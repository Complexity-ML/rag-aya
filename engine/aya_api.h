/**
 * Aya inference engine — clean C API for FFI / WASM.
 *
 * No HTTP, no platform-specific code. Link against model.c, gguf.c, quant.h.
 * INL 2025
 */
#ifndef AYA_API_H
#define AYA_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef AYA_BUILD_DLL
    #define AYA_API __declspec(dllexport)
  #else
    #define AYA_API
  #endif
#else
  #define AYA_API __attribute__((visibility("default")))
#endif

/* Opaque handle */
typedef struct aya_context aya_context;

/**
 * Load model from a GGUF file on disk (uses mmap).
 * Returns NULL on failure.
 */
AYA_API aya_context *aya_init_file(const char *gguf_path);

/**
 * Load model from an in-memory buffer (no filesystem needed — for WASM).
 * The buffer must remain valid for the lifetime of the context.
 * Returns NULL on failure.
 */
AYA_API aya_context *aya_init_buffer(const uint8_t *data, size_t size);

/**
 * Per-token streaming callback.
 * Called with each decoded token string during generation.
 * user_data is passed through from aya_generate.
 */
typedef void (*aya_token_callback)(const char *token, void *user_data);

/**
 * Generate text from a prompt.
 *
 * Applies the Cohere chat template (BOS + START_TURN + USER + ... + END_TURN
 * + START_TURN + CHATBOT), runs autoregressive decoding, and returns the
 * full response as an allocated string.
 *
 * @param ctx         Context from aya_init_file/aya_init_buffer.
 * @param prompt      User prompt (UTF-8).
 * @param max_tokens  Maximum tokens to generate.
 * @param temperature Sampling temperature (<=0 for greedy/argmax).
 * @param top_k       Top-K sampling (ignored if temperature <= 0).
 * @param cb          Optional streaming callback (NULL to disable).
 * @param user_data   Opaque pointer forwarded to cb.
 * @return            Allocated string — caller must free with aya_free_string.
 *                    Returns NULL on error.
 */
AYA_API char *aya_generate(aya_context *ctx,
                           const char *prompt,
                           int max_tokens,
                           float temperature,
                           int top_k,
                           aya_token_callback cb,
                           void *user_data);

/** Free a string returned by aya_generate. */
AYA_API void aya_free_string(char *s);

/** Free context and all associated resources. */
AYA_API void aya_free(aya_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* AYA_API_H */
