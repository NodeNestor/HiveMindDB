use crate::config::HiveMindConfig;
use crate::types::*;
use dashmap::DashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Vector embedding engine for semantic search.
///
/// Generates embeddings via external APIs (OpenAI, Ollama, CodeGate, etc.),
/// caches them in-memory, and provides cosine similarity search.
pub struct EmbeddingEngine {
    client: Client,
    config: EmbeddingConfig,
    /// memory_id → embedding vector
    vectors: DashMap<u64, Vec<f32>>,
    /// Dimensionality (set after first embedding)
    dimensions: std::sync::atomic::AtomicU32,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub dimensions: Option<u32>,
}

impl EmbeddingConfig {
    pub fn from_hivemind_config(config: &HiveMindConfig) -> Self {
        // Parse "provider:model" format, e.g. "openai:text-embedding-3-small"
        let (provider, model) = if let Some((p, m)) = config.embedding_model.split_once(':') {
            (p.to_string(), m.to_string())
        } else {
            ("openai".to_string(), config.embedding_model.clone())
        };

        let base_url = match provider.as_str() {
            "openai" => "https://api.openai.com/v1".into(),
            "ollama" => "http://localhost:11434/v1".into(),
            "codegate" => "http://localhost:9212/v1".into(),
            url if url.starts_with("http") => url.to_string(),
            _ => "https://api.openai.com/v1".into(),
        };

        let api_key = config
            .embedding_api_key
            .clone()
            .or_else(|| config.llm_api_key.clone());

        Self {
            provider,
            model,
            api_key,
            base_url,
            dimensions: None,
        }
    }
}

/// OpenAI-compatible embeddings request.
#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// OpenAI-compatible embeddings response.
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingEngine {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            client: Client::new(),
            config,
            vectors: DashMap::new(),
            dimensions: std::sync::atomic::AtomicU32::new(0),
        }
    }

    pub fn from_hivemind_config(config: &HiveMindConfig) -> Self {
        Self::new(EmbeddingConfig::from_hivemind_config(config))
    }

    /// Check if the embedding engine is configured (has an API key or is using a local provider).
    pub fn is_available(&self) -> bool {
        self.config.api_key.is_some()
            || self.config.base_url.contains("localhost")
            || self.config.base_url.contains("127.0.0.1")
    }

    /// Generate an embedding for a single text.
    pub async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding response"))
    }

    /// Generate embeddings for multiple texts in a single API call.
    pub async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/embeddings", self.config.base_url);

        let req = EmbeddingRequest {
            model: self.config.model.clone(),
            input: texts.to_vec(),
        };

        let mut builder = self.client.post(&url).json(&req);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }

        let resp = builder.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API error ({}): {}", status, body);
        }

        let emb_resp: EmbeddingResponse = resp.json().await?;

        // Track dimensions
        if let Some(first) = emb_resp.data.first() {
            self.dimensions.store(
                first.embedding.len() as u32,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        Ok(emb_resp.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Index a memory — generate and store its embedding.
    pub async fn index_memory(&self, memory: &Memory) -> anyhow::Result<()> {
        let embedding = self.embed_text(&memory.content).await?;
        self.vectors.insert(memory.id, embedding);
        debug!(memory_id = memory.id, "Memory indexed");
        Ok(())
    }

    /// Index multiple memories in a batch.
    pub async fn index_memories(&self, memories: &[Memory]) -> anyhow::Result<()> {
        if memories.is_empty() {
            return Ok(());
        }

        let texts: Vec<String> = memories.iter().map(|m| m.content.clone()).collect();
        let embeddings = self.embed_batch(&texts).await?;

        for (memory, embedding) in memories.iter().zip(embeddings) {
            self.vectors.insert(memory.id, embedding);
        }

        info!(count = memories.len(), "Batch indexed memories");
        Ok(())
    }

    /// Remove a memory's embedding from the index.
    pub fn remove_memory(&self, memory_id: u64) {
        self.vectors.remove(&memory_id);
    }

    /// Semantic search — find memories most similar to the query.
    ///
    /// Returns (memory_id, similarity_score) pairs, sorted by score descending.
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(u64, f32)>> {
        let query_embedding = self.embed_text(query).await?;
        Ok(self.search_by_vector(&query_embedding, limit))
    }

    /// Search by pre-computed vector.
    pub fn search_by_vector(
        &self,
        query_vec: &[f32],
        limit: usize,
    ) -> Vec<(u64, f32)> {
        let mut scores: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query_vec, entry.value());
                (*entry.key(), score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    /// Get number of indexed vectors.
    pub fn indexed_count(&self) -> usize {
        self.vectors.len()
    }

    /// Get the embedding dimensions (0 if no embeddings generated yet).
    pub fn dimensions(&self) -> u32 {
        self.dimensions.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if a memory has been indexed.
    pub fn is_indexed(&self, memory_id: u64) -> bool {
        self.vectors.contains_key(&memory_id)
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Hybrid scoring: combine keyword and vector scores.
pub fn hybrid_score(keyword_score: f32, vector_score: f32, vector_weight: f32) -> f32 {
    let kw_weight = 1.0 - vector_weight;
    kw_weight * keyword_score + vector_weight * vector_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_similar() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.9, 0.1];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.9);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_hybrid_score() {
        // Pure keyword
        assert_eq!(hybrid_score(0.8, 0.0, 0.0), 0.8);
        // Pure vector
        assert!((hybrid_score(0.0, 0.9, 1.0) - 0.9).abs() < 1e-6);
        // 50/50
        assert!((hybrid_score(0.6, 0.8, 0.5) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_search_by_vector() {
        let engine = EmbeddingEngine::new(EmbeddingConfig {
            provider: "test".into(),
            model: "test".into(),
            api_key: None,
            base_url: "http://localhost:1234".into(),
            dimensions: Some(3),
        });

        // Manually insert some vectors
        engine.vectors.insert(1, vec![1.0, 0.0, 0.0]);
        engine.vectors.insert(2, vec![0.0, 1.0, 0.0]);
        engine.vectors.insert(3, vec![0.9, 0.1, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let results = engine.search_by_vector(&query, 3);

        // ID 1 should be most similar (identical)
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);

        // ID 3 should be second (very similar)
        assert_eq!(results[1].0, 3);
        assert!(results[1].1 > 0.9);

        // ID 2 should be last (orthogonal)
        assert_eq!(results[2].0, 2);
        assert!(results[2].1.abs() < 1e-6);
    }

    #[test]
    fn test_embedding_config_parse() {
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "openai".into(),
            llm_api_key: Some("sk-test".into()),
            llm_model: "gpt-4o".into(),
            embedding_model: "openai:text-embedding-3-small".into(),
            embedding_api_key: None,
            data_dir: "".into(),
        };
        let ec = EmbeddingConfig::from_hivemind_config(&config);
        assert_eq!(ec.provider, "openai");
        assert_eq!(ec.model, "text-embedding-3-small");
        assert_eq!(ec.base_url, "https://api.openai.com/v1");
        // Falls back to llm_api_key
        assert_eq!(ec.api_key, Some("sk-test".into()));
    }

    #[test]
    fn test_embedding_config_ollama() {
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "ollama".into(),
            llm_api_key: None,
            llm_model: "llama3".into(),
            embedding_model: "ollama:nomic-embed-text".into(),
            embedding_api_key: None,
            data_dir: "".into(),
        };
        let ec = EmbeddingConfig::from_hivemind_config(&config);
        assert_eq!(ec.provider, "ollama");
        assert_eq!(ec.model, "nomic-embed-text");
        assert_eq!(ec.base_url, "http://localhost:11434/v1");
        assert!(ec.api_key.is_none());
    }

    #[test]
    fn test_indexed_count() {
        let engine = EmbeddingEngine::new(EmbeddingConfig {
            provider: "test".into(),
            model: "test".into(),
            api_key: None,
            base_url: "http://localhost:1234".into(),
            dimensions: None,
        });

        assert_eq!(engine.indexed_count(), 0);
        engine.vectors.insert(1, vec![1.0, 0.0]);
        assert_eq!(engine.indexed_count(), 1);
        assert!(engine.is_indexed(1));
        assert!(!engine.is_indexed(2));

        engine.remove_memory(1);
        assert_eq!(engine.indexed_count(), 0);
    }
}
