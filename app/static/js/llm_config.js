/**
 * 跨页面共享的 LLM API 配置（localStorage）。
 * 特征工程 / PU Learning / CoT 数据合成共用 finrisk_llm_* 键。
 */
(function (global) {
    'use strict';

    const LLM_STORAGE_KEYS = {
        apiKey: 'finrisk_llm_api_key',
        apiBase: 'finrisk_llm_api_base',
        model: 'finrisk_llm_model',
        serverBoot: 'finrisk_llm_server_boot_id',
    };

    function getServerBootId() {
        const meta = document.querySelector('meta[name="finrisk-server-boot-id"]');
        return meta ? (meta.getAttribute('content') || '') : '';
    }

    function clearLlmConfigStorage() {
        Object.values(LLM_STORAGE_KEYS).forEach((k) => {
            try {
                localStorage.removeItem(k);
                sessionStorage.removeItem(k);
            } catch (e) { /* private mode */ }
        });
    }

    function llmStorageRead(key) {
        try {
            return localStorage.getItem(key);
        } catch (e) { /* private mode */ }
        return null;
    }

    function llmStorageWrite(key, value) {
        try {
            if (value) localStorage.setItem(key, value);
            else localStorage.removeItem(key);
        } catch (e) { /* private mode */ }
    }

    /** 若 run.py 已重启则清空凭证；返回 true 表示本轮进程内可加载缓存 */
    function syncLlmConfigBootId() {
        const current = getServerBootId();
        if (!current) return true;
        const stored = llmStorageRead(LLM_STORAGE_KEYS.serverBoot);
        if (stored === current) return true;
        clearLlmConfigStorage();
        llmStorageWrite(LLM_STORAGE_KEYS.serverBoot, current);
        return false;
    }

    function toOpenAiV1Base(url) {
        if (!url || !String(url).trim()) return '';
        let u = String(url).trim().replace(/\/+$/, '');
        if (u.endsWith('/chat/completions')) {
            u = u.slice(0, -'/chat/completions'.length).replace(/\/+$/, '');
        }
        if (!u.endsWith('/v1')) {
            u = `${u}/v1`;
        }
        return u;
    }

    function toChatCompletionsUrl(url) {
        if (!url || !String(url).trim()) return '';
        let u = String(url).trim().replace(/\/+$/, '');
        if (u.endsWith('/chat/completions')) return u;
        if (u.endsWith('/v1')) {
            return `${u.slice(0, -'/v1'.length).replace(/\/+$/, '')}/chat/completions`;
        }
        return `${u}/chat/completions`;
    }

    /** @returns {{apiKey:string,apiBase:string,modelName:string}|null} */
    function loadLlmConfigFromStorage() {
        if (!syncLlmConfigBootId()) {
            return null;
        }
        return {
            apiKey: llmStorageRead(LLM_STORAGE_KEYS.apiKey) || '',
            apiBase: llmStorageRead(LLM_STORAGE_KEYS.apiBase) || '',
            modelName: llmStorageRead(LLM_STORAGE_KEYS.model) || '',
        };
    }

    function persistLlmConfig(cfg) {
        const apiKey = (cfg && cfg.apiKey) || '';
        const apiBase = toOpenAiV1Base((cfg && cfg.apiBase) || '');
        const modelName = (cfg && cfg.modelName) || '';
        const boot = getServerBootId();
        if (boot) llmStorageWrite(LLM_STORAGE_KEYS.serverBoot, boot);
        llmStorageWrite(LLM_STORAGE_KEYS.apiKey, apiKey.trim());
        llmStorageWrite(LLM_STORAGE_KEYS.apiBase, apiBase.trim());
        llmStorageWrite(LLM_STORAGE_KEYS.model, modelName.trim());
    }

    // 脚本加载即执行一次 boot 同步，确保服务重启后敏感缓存立即清理。
    syncLlmConfigBootId();

    global.LLM_STORAGE_KEYS = LLM_STORAGE_KEYS;
    global.getServerBootId = getServerBootId;
    global.clearLlmConfigStorage = clearLlmConfigStorage;
    global.llmStorageRead = llmStorageRead;
    global.llmStorageWrite = llmStorageWrite;
    global.syncLlmConfigBootId = syncLlmConfigBootId;
    global.toOpenAiV1Base = toOpenAiV1Base;
    global.toChatCompletionsUrl = toChatCompletionsUrl;
    global.loadLlmConfigFromStorage = loadLlmConfigFromStorage;
    global.persistLlmConfig = persistLlmConfig;
})(typeof window !== 'undefined' ? window : globalThis);
