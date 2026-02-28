use crate::config::HiveMindConfig;
use crate::types::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// LLM-powered fact extraction pipeline.
///
/// Takes conversation text and extracts structured knowledge:
/// facts, entities, relationships, and conflict resolution decisions.
///
/// Supports OpenAI-compatible APIs (OpenAI, Ollama, CodeGate, etc.)
/// and the Anthropic Messages API.
pub struct ExtractionPipeline {
    client: Client,
    config: ExtractionConfig,
}

#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub provider: String,
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: String,
}

impl ExtractionConfig {
    pub fn from_hivemind_config(config: &HiveMindConfig) -> Self {
        let (base_url, provider) = match config.llm_provider.as_str() {
            "openai" => ("https://api.openai.com/v1".into(), "openai".into()),
            "anthropic" => (
                "https://api.anthropic.com".into(),
                "anthropic".into(),
            ),
            "ollama" => (
                "http://localhost:11434/v1".into(),
                "openai".into(), // Ollama uses OpenAI-compatible API
            ),
            "codegate" => (
                "http://localhost:9212/v1".into(),
                "openai".into(), // CodeGate proxies OpenAI format
            ),
            url if url.starts_with("http") => {
                // Custom URL — assume OpenAI-compatible
                (url.into(), "openai".into())
            }
            other => {
                warn!(provider = other, "Unknown LLM provider, assuming OpenAI-compatible");
                ("https://api.openai.com/v1".into(), "openai".into())
            }
        };

        Self {
            provider,
            api_key: config.llm_api_key.clone(),
            model: config.llm_model.clone(),
            base_url,
        }
    }
}

/// Result of extracting knowledge from conversation text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub facts: Vec<ExtractedFact>,
    pub entities: Vec<ExtractedEntity>,
    pub relationships: Vec<ExtractedRelationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub content: String,
    pub memory_type: MemoryType,
    pub confidence: f32,
    pub tags: Vec<String>,
    /// How this fact relates to existing knowledge.
    pub operation: ExtractionOperation,
    /// If updating, the ID of the memory to update.
    pub updates_memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionOperation {
    Add,
    Update,
    Noop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub source_entity: String,
    pub target_entity: String,
    pub relation_type: String,
    pub description: Option<String>,
}

/// OpenAI-compatible chat completion request.
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI-compatible chat completion response.
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

/// Anthropic Messages API request.
#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ChatMessage>,
}

/// Anthropic Messages API response.
#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are a knowledge extraction engine for HiveMindDB. Your job is to extract structured knowledge from conversation text.

Given a conversation, extract:
1. **Facts**: Concrete pieces of knowledge (preferences, decisions, information)
2. **Entities**: People, projects, technologies, organizations, concepts
3. **Relationships**: How entities relate to each other

For each fact, determine:
- `operation`: "add" (new knowledge), "update" (modifies existing), "noop" (already known)
- `memory_type`: "fact" (concrete info), "episodic" (event/experience), "procedural" (how-to), "semantic" (abstract concept)
- `confidence`: 0.0-1.0 how confident you are this is accurate
- `tags`: relevant categories

Respond with ONLY valid JSON in this exact format:
{
  "facts": [
    {
      "content": "the extracted fact as a clear statement",
      "memory_type": "fact",
      "confidence": 0.95,
      "tags": ["preferences", "languages"],
      "operation": "add",
      "updates_memory_id": null
    }
  ],
  "entities": [
    {
      "name": "EntityName",
      "entity_type": "Person",
      "description": "Brief description"
    }
  ],
  "relationships": [
    {
      "source_entity": "SourceName",
      "target_entity": "TargetName",
      "relation_type": "maintains",
      "description": "Brief description"
    }
  ]
}"#;

impl ExtractionPipeline {
    pub fn new(config: ExtractionConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn from_hivemind_config(config: &HiveMindConfig) -> Self {
        Self::new(ExtractionConfig::from_hivemind_config(config))
    }

    /// Check if the pipeline is configured (has an API key or is using a local provider).
    pub fn is_available(&self) -> bool {
        self.config.api_key.is_some()
            || self.config.base_url.contains("localhost")
            || self.config.base_url.contains("127.0.0.1")
    }

    /// Extract knowledge from conversation messages.
    pub async fn extract(
        &self,
        messages: &[ConversationMessage],
        existing_memories: &[Memory],
    ) -> anyhow::Result<ExtractionResult> {
        let conversation_text = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let mut user_prompt = format!("Extract knowledge from this conversation:\n\n{}", conversation_text);

        // Include existing memories for conflict resolution
        if !existing_memories.is_empty() {
            let existing = existing_memories
                .iter()
                .take(20) // Limit context size
                .map(|m| format!("  [#{}] {}", m.id, m.content))
                .collect::<Vec<_>>()
                .join("\n");
            user_prompt.push_str(&format!(
                "\n\nExisting memories (check for conflicts/updates):\n{}",
                existing
            ));
        }

        let response_text = self.call_llm(&user_prompt).await?;
        debug!(response = %response_text, "LLM extraction response");

        // Parse JSON response — handle markdown code blocks
        let json_str = extract_json_from_response(&response_text);
        let result: ExtractionResult =
            serde_json::from_str(json_str).map_err(|e| {
                anyhow::anyhow!("Failed to parse extraction response: {} — raw: {}", e, json_str)
            })?;

        info!(
            facts = result.facts.len(),
            entities = result.entities.len(),
            relationships = result.relationships.len(),
            "Extraction complete"
        );

        Ok(result)
    }

    async fn call_llm(&self, user_prompt: &str) -> anyhow::Result<String> {
        if self.config.provider == "anthropic" {
            self.call_anthropic(user_prompt).await
        } else {
            self.call_openai_compatible(user_prompt).await
        }
    }

    async fn call_openai_compatible(&self, user_prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let req = ChatRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: EXTRACTION_SYSTEM_PROMPT.into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: user_prompt.into(),
                },
            ],
            temperature: 0.1,
            max_tokens: 4096,
            response_format: Some(ResponseFormat {
                format_type: "json_object".into(),
            }),
        };

        let mut builder = self.client.post(&url).json(&req);
        if let Some(ref key) = self.config.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }

        let resp = builder.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API error ({}): {}", status, body);
        }

        let chat_resp: ChatResponse = resp.json().await?;
        chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("Empty LLM response"))
    }

    async fn call_anthropic(&self, user_prompt: &str) -> anyhow::Result<String> {
        let url = format!("{}/v1/messages", self.config.base_url);

        let req = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: 4096,
            messages: vec![
                ChatMessage {
                    role: "user".into(),
                    content: format!("{}\n\n{}", EXTRACTION_SYSTEM_PROMPT, user_prompt),
                },
            ],
        };

        let mut builder = self.client.post(&url).json(&req);
        if let Some(ref key) = self.config.api_key {
            builder = builder
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01");
        }

        let resp = builder.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error ({}): {}", status, body);
        }

        let api_resp: AnthropicResponse = resp.json().await?;
        api_resp
            .content
            .first()
            .and_then(|c| c.text.clone())
            .ok_or_else(|| anyhow::anyhow!("Empty Anthropic response"))
    }
}

/// Strip markdown code blocks from LLM response to get raw JSON.
fn extract_json_from_response(text: &str) -> &str {
    let trimmed = text.trim();
    // Handle ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Handle ``` ... ``` blocks
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_from_response_plain() {
        let json = r#"{"facts": [], "entities": [], "relationships": []}"#;
        assert_eq!(extract_json_from_response(json), json);
    }

    #[test]
    fn test_extract_json_from_response_code_block() {
        let input = "```json\n{\"facts\": []}\n```";
        assert_eq!(extract_json_from_response(input), r#"{"facts": []}"#);
    }

    #[test]
    fn test_extract_json_from_response_bare_block() {
        let input = "```\n{\"facts\": []}\n```";
        assert_eq!(extract_json_from_response(input), r#"{"facts": []}"#);
    }

    #[test]
    fn test_extraction_config_openai() {
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "openai".into(),
            llm_api_key: Some("sk-test".into()),
            llm_model: "gpt-4o".into(),
            embedding_model: "".into(),
            embedding_api_key: None,
            data_dir: "".into(),
        };
        let ec = ExtractionConfig::from_hivemind_config(&config);
        assert_eq!(ec.base_url, "https://api.openai.com/v1");
        assert_eq!(ec.provider, "openai");
    }

    #[test]
    fn test_extraction_config_codegate() {
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "codegate".into(),
            llm_api_key: None,
            llm_model: "claude-sonnet-4-20250514".into(),
            embedding_model: "".into(),
            embedding_api_key: None,
            data_dir: "".into(),
        };
        let ec = ExtractionConfig::from_hivemind_config(&config);
        assert_eq!(ec.base_url, "http://localhost:9212/v1");
        assert_eq!(ec.provider, "openai");
    }

    #[test]
    fn test_extraction_config_custom_url() {
        let config = HiveMindConfig {
            listen_addr: "".into(),
            rtdb_url: "".into(),
            llm_provider: "http://my-proxy:8080/v1".into(),
            llm_api_key: None,
            llm_model: "llama3".into(),
            embedding_model: "".into(),
            embedding_api_key: None,
            data_dir: "".into(),
        };
        let ec = ExtractionConfig::from_hivemind_config(&config);
        assert_eq!(ec.base_url, "http://my-proxy:8080/v1");
        assert_eq!(ec.provider, "openai");
    }

    #[test]
    fn test_parse_extraction_result() {
        let json = r#"{
            "facts": [
                {
                    "content": "User prefers Rust over Python",
                    "memory_type": "fact",
                    "confidence": 0.95,
                    "tags": ["preferences", "languages"],
                    "operation": "add",
                    "updates_memory_id": null
                }
            ],
            "entities": [
                {
                    "name": "Rust",
                    "entity_type": "Language",
                    "description": "Systems programming language"
                }
            ],
            "relationships": [
                {
                    "source_entity": "User",
                    "target_entity": "Rust",
                    "relation_type": "prefers",
                    "description": null
                }
            ]
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].content, "User prefers Rust over Python");
        assert_eq!(result.facts[0].operation, ExtractionOperation::Add);
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.relationships.len(), 1);
    }

    #[test]
    fn test_pipeline_availability() {
        let pipeline = ExtractionPipeline::new(ExtractionConfig {
            provider: "openai".into(),
            api_key: Some("sk-test".into()),
            model: "gpt-4o".into(),
            base_url: "https://api.openai.com/v1".into(),
        });
        assert!(pipeline.is_available());

        let local_pipeline = ExtractionPipeline::new(ExtractionConfig {
            provider: "openai".into(),
            api_key: None,
            model: "llama3".into(),
            base_url: "http://localhost:11434/v1".into(),
        });
        assert!(local_pipeline.is_available());

        let no_key = ExtractionPipeline::new(ExtractionConfig {
            provider: "openai".into(),
            api_key: None,
            model: "gpt-4o".into(),
            base_url: "https://api.openai.com/v1".into(),
        });
        assert!(!no_key.is_available());
    }
}
