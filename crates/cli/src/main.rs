use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "hmdb", about = "HiveMindDB CLI — distributed AI agent memory")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show cluster status and memory stats
    Status {
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Add a memory
    Add {
        /// Memory content
        content: String,
        /// Agent ID
        #[arg(long)]
        agent: Option<String>,
        /// User ID
        #[arg(long)]
        user: Option<String>,
        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Memory type (fact, episodic, procedural, semantic)
        #[arg(long, default_value = "fact")]
        memory_type: String,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Search memories (hybrid: keyword + vector similarity)
    Search {
        /// Search query
        query: String,
        /// Filter by agent
        #[arg(long)]
        agent: Option<String>,
        /// Filter by user
        #[arg(long)]
        user: Option<String>,
        /// Filter by tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Max results
        #[arg(long, default_value = "10")]
        limit: usize,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Extract knowledge from conversation text using LLM
    Extract {
        /// Conversation text (or use --file for a file)
        text: Option<String>,
        /// File containing conversation (JSON array of {role, content})
        #[arg(long)]
        file: Option<String>,
        /// Agent ID
        #[arg(long)]
        agent: Option<String>,
        /// User ID
        #[arg(long)]
        user: Option<String>,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Show memory history (audit trail)
    History {
        /// Memory ID
        id: u64,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Invalidate (forget) a memory
    Forget {
        /// Memory ID
        id: u64,
        /// Reason for forgetting
        #[arg(long, default_value = "manual")]
        reason: String,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Show entity details and relationships
    Entity {
        /// Entity name
        name: String,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Graph traversal from an entity
    Traverse {
        /// Entity ID to start from
        entity_id: u64,
        /// Max traversal depth
        #[arg(long, default_value = "2")]
        depth: usize,
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// List channels
    Channels {
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// List registered agents
    Agents {
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();

    match cli.command {
        Commands::Status { addr } => {
            let resp: Value = client
                .get(format!("{}/api/v1/status", addr))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            println!("HiveMindDB Status:");
            println!("  Memories:           {}", resp["memories"]);
            println!("  Valid:              {}", resp["valid_memories"]);
            println!("  Entities:           {}", resp["entities"]);
            println!("  Relationships:      {}", resp["relationships"]);
            println!("  Episodes:           {}", resp["episodes"]);
            println!("  Agents:             {}", resp["agents"]);
            println!("  Embeddings indexed: {}", resp["embeddings_indexed"]);
            println!("  Embedding dims:     {}", resp["embedding_dimensions"]);
            println!("  Extraction:         {}", if resp["extraction_available"].as_bool().unwrap_or(false) { "available" } else { "not configured" });
            println!("  Replication:        {}", if resp["replication_enabled"].as_bool().unwrap_or(false) { "enabled" } else { "standalone" });
        }

        Commands::Add {
            content,
            agent,
            user,
            tags,
            memory_type,
            addr,
        } => {
            let tags_vec: Vec<String> = tags
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            let resp: Value = client
                .post(format!("{}/api/v1/memories", addr))
                .json(&serde_json::json!({
                    "content": content,
                    "memory_type": memory_type,
                    "agent_id": agent,
                    "user_id": user,
                    "tags": tags_vec,
                }))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            println!("Memory #{} added", resp["id"]);
        }

        Commands::Search {
            query,
            agent,
            user,
            tags,
            limit,
            addr,
        } => {
            let tags_vec: Vec<String> = tags
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            let resp: Vec<Value> = client
                .post(format!("{}/api/v1/search", addr))
                .json(&serde_json::json!({
                    "query": query,
                    "agent_id": agent,
                    "user_id": user,
                    "tags": tags_vec,
                    "limit": limit,
                }))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            if resp.is_empty() {
                println!("No memories found.");
            } else {
                println!("Found {} result(s):", resp.len());
                for result in &resp {
                    let mem = &result["memory"];
                    println!(
                        "  #{} [score: {:.2}] {}",
                        mem["id"], result["score"], mem["content"]
                    );
                    if let Some(tags) = mem["tags"].as_array() {
                        if !tags.is_empty() {
                            let tag_strs: Vec<&str> =
                                tags.iter().filter_map(|t| t.as_str()).collect();
                            println!("       tags: {}", tag_strs.join(", "));
                        }
                    }
                }
            }
        }

        Commands::Extract {
            text,
            file,
            agent,
            user,
            addr,
        } => {
            let messages: Vec<Value> = if let Some(file_path) = file {
                let content = std::fs::read_to_string(&file_path)
                    .context("Failed to read conversation file")?;
                serde_json::from_str(&content)
                    .context("File must be a JSON array of {role, content} objects")?
            } else if let Some(text) = text {
                vec![serde_json::json!({"role": "user", "content": text})]
            } else {
                anyhow::bail!("Provide conversation text or --file");
            };

            let resp: Value = client
                .post(format!("{}/api/v1/extract", addr))
                .json(&serde_json::json!({
                    "messages": messages,
                    "agent_id": agent,
                    "user_id": user,
                }))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            if let Some(added) = resp["memories_added"].as_array() {
                if !added.is_empty() {
                    println!("Added {} memories:", added.len());
                    for m in added {
                        println!("  #{}: {}", m["id"], m["content"]);
                    }
                }
            }
            if let Some(updated) = resp["memories_updated"].as_array() {
                if !updated.is_empty() {
                    println!("Updated {} memories:", updated.len());
                    for m in updated {
                        println!("  #{}: {}", m["id"], m["content"]);
                    }
                }
            }
            if let Some(entities) = resp["entities_added"].as_array() {
                if !entities.is_empty() {
                    println!("Added {} entities:", entities.len());
                    for e in entities {
                        println!("  {} ({})", e["name"], e["entity_type"]);
                    }
                }
            }
            if let Some(rels) = resp["relationships_added"].as_array() {
                if !rels.is_empty() {
                    println!("Added {} relationships", rels.len());
                }
            }
            if let Some(skipped) = resp["skipped"].as_u64() {
                if skipped > 0 {
                    println!("Skipped {} already-known facts", skipped);
                }
            }
        }

        Commands::History { id, addr } => {
            let resp: Vec<Value> = client
                .get(format!("{}/api/v1/memories/{}/history", addr, id))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            println!("History for memory #{}:", id);
            for entry in &resp {
                println!(
                    "  [{}] {} by {} — {}",
                    entry["timestamp"],
                    entry["operation"],
                    entry["changed_by"],
                    entry["reason"]
                );
                if let Some(old) = entry["old_content"].as_str() {
                    println!("    old: {}", old);
                }
                println!("    new: {}", entry["new_content"]);
            }
        }

        Commands::Forget { id, reason, addr } => {
            let resp = client
                .delete(format!("{}/api/v1/memories/{}", addr, id))
                .json(&serde_json::json!({
                    "reason": reason,
                    "changed_by": "cli",
                }))
                .send()
                .await
                .context("Failed to connect")?;

            if resp.status().is_success() {
                println!("Memory #{} invalidated", id);
            } else {
                println!("Failed: {}", resp.text().await.unwrap_or_default());
            }
        }

        Commands::Entity { name, addr } => {
            let resp: Result<Value, _> = client
                .post(format!("{}/api/v1/entities/find", addr))
                .json(&serde_json::json!({ "name": name }))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await;

            match resp {
                Ok(entity) => {
                    println!("Entity: {} ({})", entity["name"], entity["entity_type"]);
                    if let Some(desc) = entity["description"].as_str() {
                        println!("  Description: {}", desc);
                    }
                    println!("  ID: {}", entity["id"]);

                    if let Ok(rels) = client
                        .get(format!(
                            "{}/api/v1/entities/{}/relationships",
                            addr, entity["id"]
                        ))
                        .send()
                        .await
                    {
                        if let Ok(rels) = rels.json::<Vec<Value>>().await {
                            if !rels.is_empty() {
                                println!("  Relationships:");
                                for rel in &rels {
                                    println!(
                                        "    --{}-->  {} ({})",
                                        rel[0]["relation_type"],
                                        rel[1]["name"],
                                        rel[1]["entity_type"]
                                    );
                                }
                            }
                        }
                    }
                }
                Err(_) => println!("Entity '{}' not found", name),
            }
        }

        Commands::Traverse {
            entity_id,
            depth,
            addr,
        } => {
            let resp: Vec<Value> = client
                .post(format!("{}/api/v1/graph/traverse", addr))
                .json(&serde_json::json!({
                    "entity_id": entity_id,
                    "depth": depth,
                }))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            println!(
                "Graph traversal from entity #{} (depth {}):",
                entity_id, depth
            );
            for entry in &resp {
                let entity = &entry[0];
                let rels = entry[1].as_array();
                println!(
                    "  {} ({}) — {} relationship(s)",
                    entity["name"],
                    entity["entity_type"],
                    rels.map(|r| r.len()).unwrap_or(0)
                );
            }
        }

        Commands::Channels { addr } => {
            let resp: Vec<Value> = client
                .get(format!("{}/api/v1/channels", addr))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            println!("Channels:");
            if resp.is_empty() {
                println!("  (none)");
            }
            for ch in &resp {
                println!(
                    "  #{} {} ({}) — created by {}",
                    ch["id"], ch["name"], ch["channel_type"], ch["created_by"]
                );
            }
        }

        Commands::Agents { addr } => {
            let resp: Vec<Value> = client
                .get(format!("{}/api/v1/agents", addr))
                .send()
                .await
                .context("Failed to connect")?
                .json()
                .await?;

            println!("Registered Agents:");
            if resp.is_empty() {
                println!("  (none)");
            }
            for agent in &resp {
                println!(
                    "  {} ({}) — {} — {} memories",
                    agent["name"],
                    agent["agent_type"],
                    agent["status"],
                    agent["memory_count"]
                );
            }
        }
    }

    Ok(())
}
