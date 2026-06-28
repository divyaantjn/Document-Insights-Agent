import os
import json
import boto3
from botocore.exceptions import ClientError
import base64
import psycopg2
import psycopg2.pool
import psycopg2.extras
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from litellm import Router
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import sys
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

# Providers that require credentials + thinking bundled inside model_kwargs
_CHATLITELLM_KWARGS_PROVIDERS = {"bedrock", "vertex_ai", "vertex"}

# ── Router fallback policy ────────────────────────────────────────────────────
ROUTER_CONFIG = {
    "num_retries": 2,
    "timeout": 30,
    "allowed_fails": 3,
    "cooldown_time": 60,
    "retry_policy": {
        "RateLimitErrorRetries": 3,
        "TimeoutErrorRetries": 2,
        "InternalServerErrorRetries": 2,
        "AuthenticationErrorRetries": 0,
        
    }
}


class ConfigType(Enum):
    TEAM = "team"
    AGENT = "agent"


class ModelConfig:
    def __init__(self):
        self.db_pool = None
        self.router = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._executor_shutdown = False
        self._agent_id = None

        # --- LLM backend flags (set these directly on the instance) ---
        self.use_chatlitellm: bool = False
        self.use_litellm_completion: bool = True

    # ------------------------------------------------------------------ #
    #  build_llm_params  (litellm completion path only)                   #
    # ------------------------------------------------------------------ #

    def build_llm_params(
        self,
        provider: str,
        selected_model: str,
        model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the llm_params dict for litellm completion path.

        Routing:
          1. use_litellm_completion=True  -> pass everything through unchanged,
                                            no thinking params added.
          2. use_chatlitellm=True + provider in {bedrock, vertex_ai, vertex}
                                          -> all credentials AND thinking block
                                            go into model_kwargs.
          3. use_chatlitellm=True + other providers
                                          -> thinking block stays at top level.
          4. Neither flag set             -> same as (3) for backward compat.
        """
        provider_model = f"{provider}/{selected_model}"

        if self.use_litellm_completion:
            return {
                "model": provider_model,
                **model_config,
            }

        if self.use_chatlitellm:
            if provider.lower() in _CHATLITELLM_KWARGS_PROVIDERS:
                creds = {k: v for k, v in model_config.items() if k != "thinking"}
                return {
                    "model": provider_model,
                    "model_kwargs": {
                        **creds,
                        "thinking": {"type": "enabled", "budget_tokens": 1024},
                    },
                }
            else:
                return {
                    "model": provider_model,
                    **model_config,
                    "model_kwargs": {
                        "thinking": {"type": "enabled", "budget_tokens": 1024}
                    },
                }

        # Neither flag -- keep existing top-level thinking behaviour.
        return {
            "model": provider_model,
            **model_config,
            "model_kwargs": {
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            },
        }

    # ------------------------------------------------------------------ #
    #  build_litellm_router  (use_chatlitellm path only)                  #
    # ------------------------------------------------------------------ #

    def build_litellm_router(self, configs: List[Dict[str, Any]]) -> Router:
        """
        Build and return a LiteLLM Router from an ordered list of configs.

        - Single config   -> Router with one deployment, no fallback.
        - Multiple configs -> Router with priority_order ordering + automatic
                             fallback chain.

        The caller (agent code) is responsible for wrapping this Router in
        ChatLiteLLMRouter and handling any provider-specific model_kwargs.
        """
        if not configs:
            raise ValueError("Cannot build router: configs list is empty")

        model_list = []
        fallback_chain = []

        for cfg in configs:
            provider_model = f"{cfg['provider']}/{cfg['selected_model']}"
            logical_name = f"model-priority-{cfg['priority_order']}"

            model_list.append({
                "model_name": logical_name,
                "litellm_params": {
                    "model": provider_model,
                    "order": cfg["priority_order_int"],  # numeric int, e.g. 1, 2, 10
                    **cfg["config"],
                }
            })
            fallback_chain.append(logical_name)

        fallbacks = []
        if len(fallback_chain) > 1:
            fallbacks = [{fallback_chain[0]: fallback_chain[1:]}]

        router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            **ROUTER_CONFIG
        )

        logging.info(
            f"Built LiteLLM Router with {len(configs)} deployment(s): "
            + ", ".join(
                f"priority_order={c['priority_order']} "
                f"{c['provider']}/{c['selected_model']}"
                for c in configs
            )
        )

        return router

    # ------------------------------------------------------------------ #
    #  Agent ID                                                            #
    # ------------------------------------------------------------------ #

    @property
    def agent_id(self) -> str:
        if self._agent_id is None:
            self._agent_id = os.getenv("AGENT_UNIQUE_ID")
            if not self._agent_id:
                raise ValueError("AGENT_UNIQUE_ID environment variable is required")
        return self._agent_id

    # ------------------------------------------------------------------ #
    #  DB pool lifecycle                                                   #
    # ------------------------------------------------------------------ #

    async def initialize_db_pool(self):
        try:
            if self._executor_shutdown:
                self.executor = ThreadPoolExecutor(max_workers=10)
                self._executor_shutdown = False

            db_host     = os.getenv("LLM_CONFIG_DB_HOST")
            db_port     = os.getenv("LLM_CONFIG_DB_PORT")
            db_name     = os.getenv("LLM_CONFIG_DB_NAME")
            db_user     = os.getenv("LLM_CONFIG_DB_USER")
            db_password = os.getenv("LLM_CONFIG_DB_PASSWORD")

            if not all([db_host, db_port, db_name, db_user, db_password]):
                raise ValueError(
                    "All database environment variables (LLM_CONFIG_DB_HOST, "
                    "LLM_CONFIG_DB_PORT, LLM_CONFIG_DB_NAME, LLM_CONFIG_DB_USER, "
                    "LLM_CONFIG_DB_PASSWORD) are required"
                )

            loop = asyncio.get_event_loop()
            self.db_pool = await loop.run_in_executor(
                self.executor,
                lambda: psycopg2.pool.ThreadedConnectionPool(
                    1, 2,
                    host=db_host, port=db_port,
                    database=db_name, user=db_user, password=db_password,
                ),
            )
            logging.info("Database connection pool initialized")
        except Exception as e:
            logging.info(f"Failed to initialize database pool: {str(e)}")
            raise

    async def close_db_pool(self):
        if self.db_pool:
            loop = asyncio.get_event_loop()
            if not self._executor_shutdown:
                await loop.run_in_executor(self.executor, self.db_pool.closeall)
            else:
                self.db_pool.closeall()
            self.db_pool = None
            logging.info("Database connection pool closed")

    async def shutdown(self):
        await self.close_db_pool()
        if not self._executor_shutdown:
            self.executor.shutdown(wait=True)
            self._executor_shutdown = True
            logging.info("Executor shutdown complete")

    # ------------------------------------------------------------------ #
    #  DB helpers                                                          #
    # ------------------------------------------------------------------ #

    def _execute_query(self, query: str, params: tuple):
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
        finally:
            if conn:
                self.db_pool.putconn(conn)

    def _execute_query_all(self, query: str, params: tuple) -> list:
        conn = None
        try:
            conn = self.db_pool.getconn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        finally:
            if conn:
                self.db_pool.putconn(conn)

    # ------------------------------------------------------------------ #
    #  Config row parser                                                   #
    # ------------------------------------------------------------------ #

    def _parse_config_row(self, row, config_type: str) -> Dict[str, Any]:
        config = row["resolved_config"]
        if isinstance(config, str):
            config = self.decrypt(config)
            config = json.loads(config)

        raw_priority = row["priority_order"]
        if raw_priority is None:
            logger.warning(f"NULL priority_order for model {row.get('selected_model')} — defaulting to P1")
            raw_priority = "P1"
        priority_int = int(raw_priority[1:])

        return {
            "config_type": config_type,
            "provider": row["provider"],
            "selected_model": row["selected_model"],
            "config": config,
            "priority_order": raw_priority,        # kept as 'P1' for labels/logging
            "priority_order_int": priority_int,    # numeric value for Router 'order'
        }

    # ------------------------------------------------------------------ #
    #  Agent-specific configs (all priority rows)                          #
    # ------------------------------------------------------------------ #

    async def _get_agent_specific_configs(
        self, team_id: str, agent_unique_id: str
    ) -> List[Dict[str, Any]]:
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        query = """
        WITH resolved_agent AS (
            SELECT DISTINCT tlc.agent_id
            FROM teams_llm_config AS tlc
            WHERE
                tlc.is_active = TRUE
                AND tlc.team_id = %s
                AND tlc.agent_unique_id = %s
                AND tlc.config_type = 'agent'
                AND tlc.agent_id IS NOT NULL
            LIMIT 1
        )
        SELECT
            m.model_code       AS selected_model,
            p.name             AS provider,
            plc.config         AS resolved_config,
            alc.priority_order AS priority_order
        FROM
            resolved_agent        AS ra
        JOIN
            agents_llm_config     AS alc
                ON alc.agent_id = ra.agent_id
        JOIN
            llm_models            AS m
                ON m.id = alc.model_id
        JOIN
            llm_providers         AS p
                ON p.id = m.provider_id
        JOIN
            providers_llm_config  AS plc
                ON plc.id = alc.provider_config_id
            AND plc.is_active = TRUE
        WHERE
            alc.is_active = TRUE
            AND alc.provider_config_id IS NOT NULL
        ORDER BY
            CAST(NULLIF(SUBSTRING(alc.priority_order::text FROM 2), '') AS INTEGER) ASC NULLS LAST
    """

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                loop = asyncio.get_event_loop()
                rows = await loop.run_in_executor(
                    self.executor,
                    self._execute_query_all,
                    query,
                    (team_id, agent_unique_id),
                )

                if not rows:
                    logging.info(
                        f"No agent-specific config found for agent_unique_id: "
                        f"{agent_unique_id} in team: {team_id}"
                    )
                    return []

                logging.info(
                    f"Found {len(rows)} agent-specific config(s) for "
                    f"agent_unique_id: {agent_unique_id} in team: {team_id}"
                )
                return [
                    self._parse_config_row(row, ConfigType.AGENT.value)
                    for row in rows
                ]

            except psycopg2.OperationalError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.info(
                        f"Failed to fetch agent model configs after "
                        f"{max_retries} retries: {str(e)}"
                    )
                    raise
                logging.info(
                    f"Connection error fetching agent configs, "
                    f"retrying ({retry_count}/{max_retries}): {str(e)}"
                )
                await asyncio.sleep(2 ** retry_count)

            except Exception as e:
                logging.info(
                    f"Failed to fetch agent model configs for agent_unique_id "
                    f"{agent_unique_id} in team {team_id}: {str(e)}"
                )
                raise

    # ------------------------------------------------------------------ #
    #  Team-level configs (all priority rows)                              #
    # ------------------------------------------------------------------ #

    async def _get_team_level_configs(self, team_id: str) -> List[Dict[str, Any]]:
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")

        query = """
            SELECT
                m.model_code            AS selected_model,
                p.name                  AS provider,
                CASE
                    WHEN tlc.key_managed_by = 'Provider-Managed'
                         AND tlc.provider_config_id IS NOT NULL
                        THEN plc.config
                    ELSE tlc.config
                END                     AS resolved_config,
                tlc.priority_order      AS priority_order
            FROM
                teams_llm_config        AS tlc
            JOIN
                llm_models              AS m    ON m.id = tlc.model_id
            JOIN
                llm_providers           AS p    ON p.id = m.provider_id
            LEFT JOIN
                providers_llm_config    AS plc  ON plc.id = tlc.provider_config_id
                                                AND plc.is_active = TRUE
            WHERE
                tlc.is_active           = TRUE
                AND tlc.config_type     = 'team'
                AND tlc.agent_unique_id IS NULL
                AND tlc.team_id         = %s
            ORDER BY CAST(NULLIF(SUBSTRING(tlc.priority_order::text FROM 2), '') AS INTEGER) ASC NULLS LAST
        """

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                loop = asyncio.get_event_loop()
                rows = await loop.run_in_executor(
                    self.executor,
                    self._execute_query_all,
                    query,
                    (team_id,),
                )

                if not rows:
                    raise ValueError(
                        f"No model configuration found for team_id: {team_id}"
                    )

                logging.info(
                    f"Found {len(rows)} team-level config(s) for team: {team_id}"
                )
                return [
                    self._parse_config_row(row, ConfigType.TEAM.value)
                    for row in rows
                ]

            except psycopg2.OperationalError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.info(
                        f"Failed to fetch team model configs after "
                        f"{max_retries} retries: {str(e)}"
                    )
                    raise
                logging.info(
                    f"Connection error fetching team configs, "
                    f"retrying ({retry_count}/{max_retries}): {str(e)}"
                )
                await asyncio.sleep(2 ** retry_count)

            except Exception as e:
                logging.info(
                    f"Failed to fetch team model configs for team "
                    f"{team_id}: {str(e)}"
                )
                raise

    # ------------------------------------------------------------------ #
    #  Internal: always returns the full ordered list                      #
    # ------------------------------------------------------------------ #

    async def _resolve_all_configs(self, team_id: str) -> List[Dict[str, Any]]:
        """
        Core resolution -- always returns the full priority-ordered list.
        Agent-specific configs take precedence over team-level configs.
        """
        effective_agent_id = self.agent_id

        try:
            agent_configs = await self._get_agent_specific_configs(
                team_id, effective_agent_id
            )
            if agent_configs:
                logging.info(
                    f"Using {len(agent_configs)} agent-specific config(s) "
                    f"for agent: {effective_agent_id}"
                )
                return agent_configs

            logging.info(
                f"No agent-specific config found, using team-level config(s) "
                f"for team: {team_id}"
            )
            return await self._get_team_level_configs(team_id)

        except Exception as e:
            logging.info(
                f"Failed to get model configurations for team {team_id}, "
                f"agent {effective_agent_id}: {str(e)}"
            )
            raise

    # ------------------------------------------------------------------ #
    #  Main entry point -- flag-aware                                      #
    # ------------------------------------------------------------------ #

    async def get_team_model_config(
        self, team_id: str
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Flag-aware config resolver. Single entry point for all callers.

        use_chatlitellm=True:
            Returns List[Dict[str, Any]] -- all priority-ordered configs.
            Pass the list to build_litellm_router() to get a LiteLLM Router,
            then wrap it in ChatLiteLLMRouter inside agent code.

        use_litellm_completion=True (default):
            Returns Dict[str, Any] -- the single config with the lowest
            priority_order value (primary provider only, no fallback list).
            Pass directly to build_llm_params() then to litellm.acompletion().

        use_litellm_completion takes precedence if both flags are True.
        """
        all_configs = await self._resolve_all_configs(team_id)

        if self.use_litellm_completion:
            primary = all_configs[0]
            logging.info(
                f"[litellm_completion] Returning primary config: "
                f"priority_order={primary['priority_order']} "
                f"{primary['provider']}/{primary['selected_model']}"
            )
            return primary

        # ChatLiteLLM path: return full list for router construction in agent.
        logging.info(
            f"[chatlitellm] Returning {len(all_configs)} config(s) "
            f"for router construction"
        )
        return all_configs

    # ------------------------------------------------------------------ #
    #  Backward-compat: raw LiteLLM Router helpers                        #
    # ------------------------------------------------------------------ #

    def create_router_for_config(
        self, provider: str, selected_model: str, model_config: Dict[str, Any]
    ) -> Router:
        """BACKWARD COMPATIBLE: Create a raw LiteLLM Router for a single config."""
        try:
            provider_model = f"{provider}/{selected_model}"
            model_list = [{
                "model_name": selected_model,
                "litellm_params": {
                    "model": provider_model,
                    **model_config,
                },
            }]
            logging.info(f"Creating router with model list: {model_list}")
            router = Router(model_list=model_list)
            logging.info(f"Created router for model: {provider_model}")
            return router
        except Exception as e:
            logging.info(
                f"Failed to create router for {provider}/{selected_model}: {str(e)}"
            )
            raise

    async def get_router_for_team(self, team_id: str) -> Tuple[Router, str, str]:
        """BACKWARD COMPATIBLE: Returns a raw LiteLLM Router using primary config only."""
        try:
            all_configs = await self._resolve_all_configs(team_id)
            primary = all_configs[0]

            provider       = primary["provider"]
            selected_model = primary["selected_model"]
            model_cfg      = primary["config"]
            config_type    = primary["config_type"]

            router = self.create_router_for_config(provider, selected_model, model_cfg)
            logging.info(
                f"Router created using {config_type} configuration "
                f"for team: {team_id}"
            )
            return router, selected_model, config_type

        except Exception as e:
            logging.info(f"Failed to get router for team {team_id}: {str(e)}")
            raise

    # ------------------------------------------------------------------ #
    #  KMS                                                                 #
    # ------------------------------------------------------------------ #

    def create_kms_client(self):
        return boto3.client('kms', region_name=os.getenv("AWS_REGION"))

    def decrypt(self, ciphertext: str) -> str:
        try:
            kms_client = self.create_kms_client()
            key_id = os.getenv("AWS_KMS_KEY_ID_ARN")
            ciphertext_blob = base64.b64decode(ciphertext)
            decrypt_params = {'CiphertextBlob': ciphertext_blob}
            if key_id:
                decrypt_params['KeyId'] = key_id
            response = kms_client.decrypt(**decrypt_params)
            return response['Plaintext'].decode('utf-8')
        except ClientError as e:
            logging.info(f"AWS KMS Decryption Error: {e}")
            raise


# ── Global instance & context manager ────────────────────────────────────────

model_config = ModelConfig()


@asynccontextmanager
async def get_model_config():
    try:
        if not model_config.db_pool:
            await model_config.initialize_db_pool()
        yield model_config
    except Exception as e:
        logging.info(f"Error in model config context: {str(e)}")
        raise
    finally:
        pass  # Keep connection pool alive for reuse


async def initialize_config():
    await model_config.initialize_db_pool()


async def cleanup_config():
    await model_config.shutdown()