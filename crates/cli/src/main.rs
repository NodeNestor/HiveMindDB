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
        /// HiveMindDB address
        #[arg(long, default_value = "http://127.0.0.1:8100")]
        addr: String,
    },

    /// Search memories
    Search {
        /// Search query
        query: String,
        /// Filter by agent
        #[arg(long)]
        agent: Option<String>,
        /// Filter by user
        #[arg(long)]
        user: Option<String>,
        /// Max results
        #[arg(long, default_value = "10")]
        limit: usize,
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
            println!("  Memories:      {}", resp["memories"]);
            println!("  Valid:         {}", resp["valid_memories"]);
            println!("  Entities:      {}", resp["entities"]);
            println!("  Relationships: {}", resp["relationships"]);
            println!("  Episodes:      {}", resp["episodes"]);
            println!("  Agents:        {}", resp["agents"]);
        }

        Commands::Add {
            content,
            agent,
            user,
            tags,
            addr,
        } => {
            let tags_vec: Vec<String> = tags
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            let resp: Value = client
                .post(format!("{}/api/v1/memories", addr))
                .json(&serde_json::json!({
                    "content": content,
                    "memory_type": "fact",
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
            limit,
            addr,
        } => {
            let resp: Vec<Value> = client
                .post(format!("{}/api/v1/search", addr))
                .json(&serde_json::json!({
                    "query": query,
                    "agent_id": agent,
                    "user_id": user,
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

                    // Fetch relationships
                    if let Ok(rels) = client
                        .get(format!(
                            "{}/api/v1/entities/{}/relationships",
                            addr, entity["id"]
                        ))
                        .send()
                        .await
                    {
                        if let Ok(rels) = rels.json::<Vec<Value>>().await {
                            println!("  Relationships:");
                            for rel in &rels {
                                println!(
                                    "    --{}-->  {} ({})",
                                    rel[0]["relation_type"], rel[1]["name"], rel[1]["entity_type"]
                                );
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

            println!("Graph traversal from entity #{} (depth {}):", entity_id, depth);
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
                    agent["name"], agent["agent_type"], agent["status"], agent["memory_count"]
                );
            }
        }
    }

    Ok(())
}
