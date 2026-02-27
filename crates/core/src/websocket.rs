use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::channels::ChannelHub;
use crate::types::*;

/// Handle a WebSocket client connection.
///
/// Clients send JSON messages to subscribe/unsubscribe to channels
/// and receive real-time updates when memories or entities change.
pub async fn handle_ws_connection(ws: WebSocket, channels: Arc<ChannelHub>) {
    let (mut ws_tx, mut ws_rx) = ws.split();
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Track active channel subscriptions for this connection
    let active_receivers: Arc<tokio::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::new()));

    info!("WebSocket client connected");

    // Task: forward internal messages to the WebSocket
    let forward_task = tokio::spawn(async move {
        while let Some(msg) = internal_rx.recv().await {
            if ws_tx.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Process incoming messages
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => {
                let text_str: &str = &text;
                match serde_json::from_str::<WsClientMessage>(text_str) {
                    Ok(client_msg) => {
                        handle_client_message(
                            client_msg,
                            &channels,
                            &internal_tx,
                            &active_receivers,
                        )
                        .await;
                    }
                    Err(e) => {
                        let err = WsServerMessage::Error {
                            message: format!("Invalid message: {}", e),
                        };
                        let _ = internal_tx.send(serde_json::to_string(&err).unwrap());
                    }
                }
            }
            Message::Ping(data) => {
                let _ = internal_tx.send(
                    serde_json::to_string(&WsServerMessage::Pong).unwrap(),
                );
                debug!("Ping received, pong sent");
                let _ = data; // Acknowledge the ping data
            }
            Message::Close(_) => {
                info!("WebSocket client disconnecting");
                break;
            }
            _ => {}
        }
    }

    // Clean up: abort all subscription receiver tasks
    let receivers = active_receivers.lock().await;
    for handle in receivers.iter() {
        handle.abort();
    }
    forward_task.abort();

    info!("WebSocket client disconnected");
}

async fn handle_client_message(
    msg: WsClientMessage,
    channels: &Arc<ChannelHub>,
    tx: &tokio::sync::mpsc::UnboundedSender<String>,
    active_receivers: &Arc<tokio::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
) {
    match msg {
        WsClientMessage::Subscribe {
            channels: channel_names,
            agent_id,
        } => {
            let mut subscribed = Vec::new();

            for channel_name in &channel_names {
                // Auto-create channels if they don't exist
                if channels.get_channel_by_name(channel_name).is_none() {
                    channels.create_channel(CreateChannelRequest {
                        name: channel_name.clone(),
                        description: None,
                        channel_type: ChannelType::Public,
                        created_by: agent_id
                            .clone()
                            .unwrap_or_else(|| "ws-client".into()),
                    });
                }

                let aid = agent_id.clone().unwrap_or_else(|| "anonymous".into());
                if let Some(rx) = channels.subscribe_by_name(channel_name, &aid) {
                    subscribed.push(channel_name.clone());
                    // Spawn a task to forward channel messages to this client
                    let tx_clone = tx.clone();
                    let handle = tokio::spawn(forward_channel_messages(rx, tx_clone));
                    active_receivers.lock().await.push(handle);
                }
            }

            let resp = WsServerMessage::Subscribed {
                channels: subscribed,
            };
            let _ = tx.send(serde_json::to_string(&resp).unwrap());
        }

        WsClientMessage::Unsubscribe { channels: _ } => {
            // Unsubscribe is handled by dropping receivers on disconnect.
            // Per-channel unsubscribe would need receiver tracking by channel name.
            debug!("Unsubscribe received (channels cleaned up on disconnect)");
        }

        WsClientMessage::Ping => {
            let _ = tx.send(serde_json::to_string(&WsServerMessage::Pong).unwrap());
        }
    }
}

async fn forward_channel_messages(
    mut rx: broadcast::Receiver<WsServerMessage>,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
) {
    loop {
        match rx.recv().await {
            Ok(msg) => {
                if let Ok(json) = serde_json::to_string(&msg) {
                    if tx.send(json).is_err() {
                        break; // Client disconnected
                    }
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(skipped = n, "Channel subscriber lagged, skipped messages");
            }
            Err(broadcast::error::RecvError::Closed) => {
                break; // Channel was dropped
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_client_message_parse_subscribe() {
        let json = r#"{"type":"subscribe","channels":["global","user:alice"],"agent_id":"agent-1"}"#;
        let msg: WsClientMessage = serde_json::from_str(json).unwrap();
        match msg {
            WsClientMessage::Subscribe {
                channels,
                agent_id,
            } => {
                assert_eq!(channels, vec!["global", "user:alice"]);
                assert_eq!(agent_id, Some("agent-1".to_string()));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_ws_client_message_parse_ping() {
        let json = r#"{"type":"ping"}"#;
        let msg: WsClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, WsClientMessage::Ping));
    }

    #[test]
    fn test_ws_server_message_serialize() {
        let msg = WsServerMessage::MemoryAdded {
            channel: "global".into(),
            memory: Memory {
                id: 1,
                content: "test".into(),
                memory_type: MemoryType::Fact,
                agent_id: None,
                user_id: None,
                session_id: None,
                confidence: 1.0,
                tags: vec![],
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                valid_from: chrono::Utc::now(),
                valid_until: None,
                source: "test".into(),
                metadata: serde_json::Value::Null,
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"memory_added\""));
        assert!(json.contains("\"channel\":\"global\""));
    }

    #[test]
    fn test_ws_server_message_subscribed() {
        let msg = WsServerMessage::Subscribed {
            channels: vec!["a".into(), "b".into()],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"subscribed\""));
    }
}
