use crate::config::HiveMindConfig;
use crate::types::*;
use dashmap::DashMap;
use rayon::prelude::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Vector embedding engine for semantic search.
///
/// Supports two backends:
/// - **Local** (default): In-process ONNX model via `fastembed` — no external API needed.
///   Uses all-MiniLM-L6-v2 (22M params, 384 dims) by default. CPU-only, ~22MB model.
/// - **API**: External embedding APIs (OpenAI, Ollama, CodeGate, etc.)
///
/// Uses a pool of model instances for concurrent embedding generation,
/// and rayon parallel iterators for fast cosine similarity search.
pub struct EmbeddingEngine {
    client: Client,
    config: EmbeddingConfig,
    /// memory_id → embedding vector
    vectors: DashMap<u64, Vec<f32>>,
    /// Dimensionality (set after first embedding)
    dimensions: std::sync::atomic::AtomicU32,
    /// Pool of local ONNX embedding models for concurrent embedding.
    /// Wrapped in Arc<Mutex> so they can be sent to spawn_blocking.
    #[cfg(feature = "local-embeddings")]
    model_pool: Vec<Arc<std::sync::Mutex<fastembed::TextEmbedding>>>,
    /// Round-robin counter for pool selection
    #[cfg(feature = "local-embeddings")]
    pool_counter: std::sync::atomic::AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub dimensions: Option<u32>,
    pub cache_dir: Option<String>,
    /// Number of model instances in the pool (default: num_cpus / 2, min 2)
    pub pool_size: usize,
}

impl EmbeddingConfig {
    pub fn from_hivemind_config(config: &HiveMindConfig) -> Self {
        // Parse "provider:model" format, e.g. "openai:text-embedding-3-small"
        // or "local:all-MiniLM-L6-v2"
        let (provider, model) = if let Some((p, m)) = config.embedding_model.split_once(':') {
            (p.to_string(), m.to_string())
        } else {
            // No prefix — default to "local" with the value as the model name
            #[cfg(feature = "local-embeddings")]
            {
                ("local".to_string(), config.embedding_model.clone())
            }
            #[cfg(not(feature = "local-embeddings"))]
            {
                ("openai".to_string(), config.embedding_model.clone())
            }
        };

        let base_url = match provider.as_str() {
            "local" => String::new(),
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

        // Pool size: half the CPU cores, minimum 2, maximum 8
        let pool_size = (num_cpus() / 2).max(2).min(8);

        Self {
            provider,
            model,
            api_key,
            base_url,
            dimensions: None,
            cache_dir: Some(format!("{}/embeddings", config.data_dir)),
            pool_size,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
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

/// Map user-friendly model names to fastembed enum variants.
#[cfg(feature = "local-embeddings")]
fn resolve_local_model(name: &str) -> fastembed::EmbeddingModel {
    match name {
        // MiniLM family (tiny, fast, great quality for size)
        "all-MiniLM-L6-v2" | "AllMiniLML6V2" | "minilm" => {
            fastembed::EmbeddingModel::AllMiniLML6V2
        }
        "all-MiniLM-L6-v2-q" | "AllMiniLML6V2Q" | "minilm-q" => {
            fastembed::EmbeddingModel::AllMiniLML6V2Q
        }
        "all-MiniLM-L12-v2" | "AllMiniLML12V2" => fastembed::EmbeddingModel::AllMiniLML12V2,
        "all-MiniLM-L12-v2-q" | "AllMiniLML12V2Q" => fastembed::EmbeddingModel::AllMiniLML12V2Q,

        // BGE family
        "bge-small-en-v1.5" | "BGESmallENV15" | "bge-small" => {
            fastembed::EmbeddingModel::BGESmallENV15
        }
        "bge-base-en-v1.5" | "BGEBaseENV15" | "bge-base" => {
            fastembed::EmbeddingModel::BGEBaseENV15
        }
        "bge-large-en-v1.5" | "BGELargeENV15" | "bge-large" => {
            fastembed::EmbeddingModel::BGELargeENV15
        }
        "bge-m3" | "BGEM3" => fastembed::EmbeddingModel::BGEM3,

        // Snowflake Arctic (small + fast)
        "snowflake-arctic-embed-xs" | "arctic-xs" => {
            fastembed::EmbeddingModel::SnowflakeArcticEmbedXS
        }
        "snowflake-arctic-embed-s" | "arctic-s" => {
            fastembed::EmbeddingModel::SnowflakeArcticEmbedS
        }
        "snowflake-arctic-embed-m" | "arctic-m" => {
            fastembed::EmbeddingModel::SnowflakeArcticEmbedM
        }

        // Nomic
        "nomic-embed-text-v1.5" | "nomic-v1.5" => fastembed::EmbeddingModel::NomicEmbedTextV15,
        "nomic-embed-text-v1" | "nomic-v1" => fastembed::EmbeddingModel::NomicEmbedTextV1,

        // GTE
        "gte-base-en-v1.5" | "gte-base" => fastembed::EmbeddingModel::GTEBaseENV15,
        "gte-large-en-v1.5" | "gte-large" => fastembed::EmbeddingModel::GTELargeENV15,

        // Multilingual E5
        "multilingual-e5-small" | "e5-small" => fastembed::EmbeddingModel::MultilingualE5Small,
        "multilingual-e5-base" | "e5-base" => fastembed::EmbeddingModel::MultilingualE5Base,

        // Jina (code-aware)
        "jina-embeddings-v2-base-code" | "jina-code" => {
            fastembed::EmbeddingModel::JinaEmbeddingsV2BaseCode
        }
        "jina-embeddings-v2-base-en" | "jina-en" => {
            fastembed::EmbeddingModel::JinaEmbeddingsV2BaseEN
        }

        // Google EmbeddingGemma
        "embedding-gemma-300m" | "gemma-300m" => fastembed::EmbeddingModel::EmbeddingGemma300M,

        // Default: all-MiniLM-L6-v2 (22M params, 384 dims — best balance)
        _ => {
            warn!(
                model = name,
                "Unknown local embedding model, defaulting to all-MiniLM-L6-v2"
            );
            fastembed::EmbeddingModel::AllMiniLML6V2
        }
    }
}

/// Create a single fastembed model instance.
#[cfg(feature = "local-embeddings")]
fn create_model_instance(
    model_name: &str,
    cache_dir: &Option<String>,
    show_progress: bool,
) -> Option<fastembed::TextEmbedding> {
    let model_enum = resolve_local_model(model_name);
    let mut init = fastembed::InitOptions::new(model_enum).with_show_download_progress(show_progress);
    if let Some(cache_dir) = cache_dir {
        init = init.with_cache_dir(std::path::PathBuf::from(cache_dir));
    }
    match fastembed::TextEmbedding::try_new(init) {
        Ok(model) => Some(model),
        Err(e) => {
            warn!(error = %e, "Failed to create model instance");
            None
        }
    }
}

impl EmbeddingEngine {
    pub fn new(config: EmbeddingConfig) -> Self {
        #[cfg(feature = "local-embeddings")]
        let model_pool = if config.provider == "local" {
            let pool_size = config.pool_size;
            info!(
                model = %config.model,
                pool_size,
                "Loading local embedding model pool (in-process ONNX, CPU-only)"
            );

            let mut pool = Vec::with_capacity(pool_size);
            for i in 0..pool_size {
                if let Some(model) =
                    create_model_instance(&config.model, &config.cache_dir, i == 0)
                {
                    pool.push(Arc::new(std::sync::Mutex::new(model)));
                }
            }

            if pool.is_empty() {
                warn!("No embedding model instances created — embeddings disabled");
            } else {
                info!(count = pool.len(), "Embedding model pool ready");
            }
            pool
        } else {
            Vec::new()
        };

        Self {
            client: Client::new(),
            config,
            vectors: DashMap::new(),
            dimensions: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "local-embeddings")]
            model_pool,
            #[cfg(feature = "local-embeddings")]
            pool_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn from_hivemind_config(config: &HiveMindConfig) -> Self {
        Self::new(EmbeddingConfig::from_hivemind_config(config))
    }

    /// Check if the embedding engine is configured and ready.
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "local-embeddings")]
        if !self.model_pool.is_empty() {
            return true;
        }

        self.config.api_key.is_some()
            || self.config.base_url.contains("localhost")
            || self.config.base_url.contains("127.0.0.1")
    }

    /// Pick the next model from the pool (round-robin).
    #[cfg(feature = "local-embeddings")]
    fn next_model_index(&self) -> usize {
        let idx = self
            .pool_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        idx % self.model_pool.len()
    }

    /// Generate an embedding for a single text.
    pub async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding response"))
    }

    /// Generate embeddings for multiple texts.
    ///
    /// Uses local ONNX model pool when available, otherwise falls back to external API.
    pub async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Try local model pool first
        #[cfg(feature = "local-embeddings")]
        if !self.model_pool.is_empty() {
            return self.embed_local_pooled(texts).await;
        }

        // Fall back to external API
        self.embed_api(texts).await
    }

    /// Embed using a model from the pool (round-robin selection).
    #[cfg(feature = "local-embeddings")]
    async fn embed_local_pooled(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let idx = self.next_model_index();
        let pool_len = self.model_pool.len();
        let texts = texts.to_vec();

        // Try to find an unlocked model first (opportunistic, avoids blocking)
        for offset in 0..pool_len {
            let try_idx = (idx + offset) % pool_len;
            if let Ok(mut model) = self.model_pool[try_idx].try_lock() {
                let embeddings = model
                    .embed(texts, None)
                    .map_err(|e| anyhow::anyhow!("Local embedding failed: {}", e))?;

                if let Some(first) = embeddings.first() {
                    self.dimensions.store(
                        first.len() as u32,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                return Ok(embeddings);
            }
        }

        // All models busy — clone Arc and wait in spawn_blocking
        let model_arc = self.model_pool[idx % pool_len].clone();

        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model = model_arc
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
            model
                .embed(texts, None)
                .map_err(|e| anyhow::anyhow!("Local embedding failed: {}", e))
        })
        .await??;

        if let Some(first) = embeddings.first() {
            self.dimensions.store(
                first.len() as u32,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        Ok(embeddings)
    }

    /// Embed using external OpenAI-compatible API.
    async fn embed_api(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
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

    /// Search by pre-computed vector using rayon parallel iteration.
    pub fn search_by_vector(
        &self,
        query_vec: &[f32],
        limit: usize,
    ) -> Vec<(u64, f32)> {
        // Collect keys+vectors into a vec for rayon (DashMap iter isn't Send)
        let entries: Vec<(u64, Vec<f32>)> = self
            .vectors
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();

        // Parallel cosine similarity computation
        let mut scores: Vec<(u64, f32)> = entries
            .par_iter()
            .map(|(id, vec)| (*id, cosine_similarity(query_vec, vec)))
            .collect();

        // Partial sort: only need top `limit` results
        if scores.len() > limit {
            scores.select_nth_unstable_by(limit, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scores.truncate(limit);
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

    /// Get the provider name for status reporting.
    pub fn provider(&self) -> &str {
        &self.config.provider
    }

    /// Get the model name for status reporting.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Get pool size for status reporting.
    pub fn pool_size(&self) -> usize {
        #[cfg(feature = "local-embeddings")]
        {
            self.model_pool.len()
        }
        #[cfg(not(feature = "local-embeddings"))]
        {
            0
        }
    }
}

/// Cosine similarity between two vectors.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // Process in chunks of 4 for better auto-vectorization
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let a0 = a[base];
        let a1 = a[base + 1];
        let a2 = a[base + 2];
        let a3 = a[base + 3];
        let b0 = b[base];
        let b1 = b[base + 1];
        let b2 = b[base + 2];
        let b3 = b[base + 3];

        dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let x = a[base + i];
        let y = b[base + i];
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
    fn test_cosine_similarity_chunked_matches_simple() {
        // Test with exactly 384 dims (like all-MiniLM-L6-v2)
        let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01 + 0.5).sin()).collect();

        let chunked = cosine_similarity(&a, &b);

        // Compute reference without chunking
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let reference = dot / (na * nb);

        assert!(
            (chunked - reference).abs() < 1e-5,
            "chunked={} reference={}",
            chunked,
            reference
        );
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
            cache_dir: None,
            pool_size: 0,
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
            data_dir: "./data".into(),
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
            data_dir: "./data".into(),
        };
        let ec = EmbeddingConfig::from_hivemind_config(&config);
        assert_eq!(ec.provider, "ollama");
        assert_eq!(ec.model, "nomic-embed-text");
        assert_eq!(ec.base_url, "http://localhost:11434/v1");
        assert!(ec.api_key.is_none());
    }

    #[test]
    fn test_embedding_config_local_explicit() {
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "openai".into(),
            llm_api_key: None,
            llm_model: "gpt-4o".into(),
            embedding_model: "local:all-MiniLM-L6-v2".into(),
            embedding_api_key: None,
            data_dir: "/data".into(),
        };
        let ec = EmbeddingConfig::from_hivemind_config(&config);
        assert_eq!(ec.provider, "local");
        assert_eq!(ec.model, "all-MiniLM-L6-v2");
        assert_eq!(ec.base_url, "");
        assert_eq!(ec.cache_dir, Some("/data/embeddings".to_string()));
    }

    #[cfg(feature = "local-embeddings")]
    #[test]
    fn test_embedding_config_local_default() {
        // No provider prefix → defaults to "local" when feature is enabled
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "openai".into(),
            llm_api_key: None,
            llm_model: "gpt-4o".into(),
            embedding_model: "all-MiniLM-L6-v2".into(),
            embedding_api_key: None,
            data_dir: "./data".into(),
        };
        let ec = EmbeddingConfig::from_hivemind_config(&config);
        assert_eq!(ec.provider, "local");
        assert_eq!(ec.model, "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_indexed_count() {
        let engine = EmbeddingEngine::new(EmbeddingConfig {
            provider: "test".into(),
            model: "test".into(),
            api_key: None,
            base_url: "http://localhost:1234".into(),
            dimensions: None,
            cache_dir: None,
            pool_size: 0,
        });

        assert_eq!(engine.indexed_count(), 0);
        engine.vectors.insert(1, vec![1.0, 0.0]);
        assert_eq!(engine.indexed_count(), 1);
        assert!(engine.is_indexed(1));
        assert!(!engine.is_indexed(2));

        engine.remove_memory(1);
        assert_eq!(engine.indexed_count(), 0);
    }

    #[cfg(feature = "local-embeddings")]
    #[test]
    fn test_resolve_local_model_variants() {
        use fastembed::EmbeddingModel;

        // Known names
        assert!(matches!(
            resolve_local_model("all-MiniLM-L6-v2"),
            EmbeddingModel::AllMiniLML6V2
        ));
        assert!(matches!(
            resolve_local_model("minilm"),
            EmbeddingModel::AllMiniLML6V2
        ));
        assert!(matches!(
            resolve_local_model("bge-small"),
            EmbeddingModel::BGESmallENV15
        ));
        assert!(matches!(
            resolve_local_model("jina-code"),
            EmbeddingModel::JinaEmbeddingsV2BaseCode
        ));

        // Unknown → default
        assert!(matches!(
            resolve_local_model("nonexistent-model"),
            EmbeddingModel::AllMiniLML6V2
        ));
    }
}
