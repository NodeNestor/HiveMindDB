#[derive(Clone, Debug)]
pub struct HiveMindConfig {
    pub listen_addr: String,
    pub rtdb_url: String,
    pub llm_provider: String,
    pub llm_api_key: Option<String>,
    pub llm_model: String,
    pub embedding_model: String,
    pub embedding_api_key: Option<String>,
    pub data_dir: String,
}
