use crate::types::*;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;
use tracing::info;

/// Manages hivemind channels — pub/sub for real-time memory sharing between agents.
pub struct ChannelHub {
    channels: DashMap<u64, Channel>,
    channel_by_name: DashMap<String, u64>,
    subscriptions: DashMap<u64, Vec<String>>, // channel_id -> [agent_ids]
    /// Broadcast senders per channel for WebSocket push.
    senders: DashMap<u64, broadcast::Sender<WsServerMessage>>,
    next_id: AtomicU64,
}

impl ChannelHub {
    pub fn new() -> Self {
        Self {
            channels: DashMap::new(),
            channel_by_name: DashMap::new(),
            subscriptions: DashMap::new(),
            senders: DashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn create_channel(&self, req: CreateChannelRequest) -> Channel {
        // Return existing channel if name already taken
        if let Some(id) = self.channel_by_name.get(&req.name) {
            if let Some(ch) = self.channels.get(id.value()) {
                return ch.clone();
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let now = chrono::Utc::now();

        let channel = Channel {
            id,
            name: req.name.clone(),
            description: req.description,
            channel_type: req.channel_type,
            created_by: req.created_by,
            created_at: now,
        };

        let (tx, _) = broadcast::channel(256);
        self.senders.insert(id, tx);
        self.channel_by_name.insert(req.name.clone(), id);
        self.channels.insert(id, channel.clone());

        info!(id, name = %channel.name, "Channel created");
        channel
    }

    pub fn get_channel(&self, id: u64) -> Option<Channel> {
        self.channels.get(&id).map(|c| c.clone())
    }

    pub fn get_channel_by_name(&self, name: &str) -> Option<Channel> {
        let id = self.channel_by_name.get(name)?;
        self.channels.get(id.value()).map(|c| c.clone())
    }

    pub fn list_channels(&self) -> Vec<Channel> {
        self.channels.iter().map(|c| c.value().clone()).collect()
    }

    pub fn subscribe(&self, channel_id: u64, agent_id: &str) -> Option<broadcast::Receiver<WsServerMessage>> {
        let sender = self.senders.get(&channel_id)?;
        self.subscriptions
            .entry(channel_id)
            .or_default()
            .push(agent_id.to_string());
        info!(channel_id, agent_id, "Agent subscribed to channel");
        Some(sender.subscribe())
    }

    pub fn subscribe_by_name(&self, channel_name: &str, agent_id: &str) -> Option<broadcast::Receiver<WsServerMessage>> {
        let id = *self.channel_by_name.get(channel_name)?;
        self.subscribe(id, agent_id)
    }

    pub fn broadcast_to_channel(&self, channel_id: u64, message: WsServerMessage) {
        if let Some(sender) = self.senders.get(&channel_id) {
            // Ignore send errors (no subscribers)
            let _ = sender.send(message);
        }
    }

    pub fn broadcast_to_channel_by_name(&self, channel_name: &str, message: WsServerMessage) {
        if let Some(id) = self.channel_by_name.get(channel_name) {
            self.broadcast_to_channel(*id, message);
        }
    }

    pub fn get_subscribers(&self, channel_id: u64) -> Vec<String> {
        self.subscriptions
            .get(&channel_id)
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_channel() {
        let hub = ChannelHub::new();
        let ch = hub.create_channel(CreateChannelRequest {
            name: "global".into(),
            description: Some("Global channel".into()),
            channel_type: ChannelType::Public,
            created_by: "system".into(),
        });
        assert_eq!(ch.name, "global");

        let channels = hub.list_channels();
        assert_eq!(channels.len(), 1);
    }

    #[test]
    fn test_create_duplicate_channel_returns_existing() {
        let hub = ChannelHub::new();
        let ch1 = hub.create_channel(CreateChannelRequest {
            name: "global".into(),
            description: None,
            channel_type: ChannelType::Public,
            created_by: "agent-1".into(),
        });
        let ch2 = hub.create_channel(CreateChannelRequest {
            name: "global".into(),
            description: Some("Different desc".into()),
            channel_type: ChannelType::Public,
            created_by: "agent-2".into(),
        });
        assert_eq!(ch1.id, ch2.id);
        assert_eq!(hub.list_channels().len(), 1);
    }

    #[test]
    fn test_subscribe_and_broadcast() {
        let hub = ChannelHub::new();
        let ch = hub.create_channel(CreateChannelRequest {
            name: "test".into(),
            description: None,
            channel_type: ChannelType::Public,
            created_by: "system".into(),
        });

        let mut rx = hub.subscribe(ch.id, "agent-1").unwrap();

        let msg = WsServerMessage::MemoryInvalidated {
            channel: "test".into(),
            memory_id: 42,
            reason: "test".into(),
        };
        hub.broadcast_to_channel(ch.id, msg);

        // Receiver should get the message
        let received = rx.try_recv().unwrap();
        match received {
            WsServerMessage::MemoryInvalidated { memory_id, .. } => {
                assert_eq!(memory_id, 42);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_get_channel_by_name() {
        let hub = ChannelHub::new();
        hub.create_channel(CreateChannelRequest {
            name: "project:rafttimedb".into(),
            description: None,
            channel_type: ChannelType::Public,
            created_by: "system".into(),
        });

        let found = hub.get_channel_by_name("project:rafttimedb").unwrap();
        assert_eq!(found.name, "project:rafttimedb");

        assert!(hub.get_channel_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_subscribers_list() {
        let hub = ChannelHub::new();
        let ch = hub.create_channel(CreateChannelRequest {
            name: "test".into(),
            description: None,
            channel_type: ChannelType::Public,
            created_by: "system".into(),
        });

        hub.subscribe(ch.id, "agent-1");
        hub.subscribe(ch.id, "agent-2");

        let subs = hub.get_subscribers(ch.id);
        assert_eq!(subs.len(), 2);
        assert!(subs.contains(&"agent-1".to_string()));
        assert!(subs.contains(&"agent-2".to_string()));
    }
}
