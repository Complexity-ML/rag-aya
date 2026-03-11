/**
 * GGUF file reader — mmap based, with buffer-based fallback for WASM.
 * INL 2025
 */
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* GGUF value types */
enum gguf_vtype {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* Read helpers — little endian from buffer */
typedef struct { uint8_t *p; } reader_t;

static uint8_t  rd_u8(reader_t *r)  { uint8_t  v = *r->p; r->p += 1; return v; }
static uint16_t rd_u16(reader_t *r) { uint16_t v; memcpy(&v, r->p, 2); r->p += 2; return v; }
static uint32_t rd_u32(reader_t *r) { uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v; }
static int32_t  rd_i32(reader_t *r) { int32_t  v; memcpy(&v, r->p, 4); r->p += 4; return v; }
static uint64_t rd_u64(reader_t *r) { uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v; }
static float    rd_f32(reader_t *r) { float    v; memcpy(&v, r->p, 4); r->p += 4; return v; }
static double   rd_f64(reader_t *r) { double   v; memcpy(&v, r->p, 8); r->p += 8; return v; }

static char *rd_string(reader_t *r) {
    uint64_t len = rd_u64(r);
    char *s = malloc(len + 1);
    memcpy(s, r->p, len);
    s[len] = '\0';
    r->p += len;
    return s;
}

/* Skip a GGUF value */
static void skip_value(reader_t *r, int type);

static void skip_value(reader_t *r, int type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: r->p += 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: r->p += 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: r->p += 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: r->p += 8; break;
        case GGUF_TYPE_STRING: { uint64_t len = rd_u64(r); r->p += len; } break;
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = rd_u32(r);
            uint64_t count = rd_u64(r);
            for (uint64_t i = 0; i < count; i++) skip_value(r, elem_type);
        } break;
    }
}

/* Read string array value */
static char **read_string_array(reader_t *r, uint64_t count) {
    char **arr = malloc(count * sizeof(char *));
    for (uint64_t i = 0; i < count; i++) {
        arr[i] = rd_string(r);
    }
    return arr;
}

static float *read_float_array(reader_t *r, uint64_t count) {
    float *arr = malloc(count * sizeof(float));
    for (uint64_t i = 0; i < count; i++) {
        arr[i] = rd_f32(r);
    }
    return arr;
}

/* Shared GGUF parser — operates on a data buffer already in memory.
 * The caller sets f->data, f->file_size, and f->owns_data before calling. */
static int gguf_parse(gguf_file *f) {
    reader_t r = { .p = f->data };

    /* Magic + version */
    uint32_t magic = rd_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Not a GGUF file (magic: 0x%08x)\n", magic);
        return 0;
    }
    uint32_t version = rd_u32(&r);
    uint64_t n_tensors = rd_u64(&r);
    uint64_t n_metadata = rd_u64(&r);

    printf("GGUF v%d: %llu tensors, %llu metadata entries\n",
           version, (unsigned long long)n_tensors, (unsigned long long)n_metadata);

    /* Parse metadata */
    for (uint64_t i = 0; i < n_metadata; i++) {
        char *key = rd_string(&r);
        uint32_t vtype = rd_u32(&r);

        /* Strip architecture prefix to match keys generically */
        const char *suffix = key;
        if      (strncmp(key, "llama.", 6) == 0)   suffix = key + 6;
        else if (strncmp(key, "cohere2.", 8) == 0)  suffix = key + 8;
        else if (strncmp(key, "cohere.", 7) == 0)   suffix = key + 7;

        if (strcmp(suffix, "embedding_length") == 0) {
            f->hidden_size = (int)rd_u32(&r);
        } else if (strcmp(suffix, "block_count") == 0) {
            f->num_layers = (int)rd_u32(&r);
        } else if (strcmp(suffix, "attention.head_count") == 0) {
            f->num_heads = (int)rd_u32(&r);
        } else if (strcmp(suffix, "attention.head_count_kv") == 0) {
            f->num_kv_heads = (int)rd_u32(&r);
        } else if (strcmp(suffix, "feed_forward_length") == 0) {
            f->intermediate_size = (int)rd_u32(&r);
        } else if (strcmp(suffix, "context_length") == 0) {
            f->context_length = (int)rd_u32(&r);
        } else if (strcmp(suffix, "rope.freq_base") == 0) {
            f->rope_theta = rd_f32(&r);
        } else if (strcmp(suffix, "logit_scale") == 0) {
            f->logit_scale = rd_f32(&r);
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            f->bos_id = (int)rd_u32(&r);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            f->eos_id = (int)rd_u32(&r);
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            uint32_t elem_type = rd_u32(&r); (void)elem_type;
            uint64_t count = rd_u64(&r);
            f->vocab_size = (int)count;
            f->vocab_tokens = read_string_array(&r, count);
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0) {
            uint32_t elem_type = rd_u32(&r); (void)elem_type;
            uint64_t count = rd_u64(&r);
            f->vocab_scores = read_float_array(&r, count);
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0) {
            uint32_t elem_type = rd_u32(&r); (void)elem_type;
            uint64_t count = rd_u64(&r);
            f->n_merges = (int)count;
            f->merges = read_string_array(&r, count);
        } else {
            skip_value(&r, vtype);
        }
        free(key);
    }

    if (f->num_kv_heads == 0) f->num_kv_heads = f->num_heads;

    /* Parse tensor infos */
    f->n_tensors = (int)n_tensors;
    for (uint64_t i = 0; i < n_tensors && i < MAX_TENSORS; i++) {
        char *name = rd_string(&r);
        strncpy(f->tensors[i].name, name, MAX_NAME_LEN - 1);
        free(name);

        uint32_t n_dims = rd_u32(&r);
        f->tensors[i].n_dims = n_dims;
        int64_t n_el = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            f->tensors[i].shape[d] = (int64_t)rd_u64(&r);
            n_el *= f->tensors[i].shape[d];
        }
        f->tensors[i].n_elements = n_el;
        f->tensors[i].type = (int)rd_u32(&r);
        f->tensors[i].offset = rd_u64(&r);

        int gs = GGML_GROUP_SIZE[f->tensors[i].type];
        int bs = GGML_BLOCK_SIZE[f->tensors[i].type];
        f->tensors[i].data_size = ((n_el + gs - 1) / gs) * bs;
    }

    /* Compute tensor data offset (aligned) */
    size_t cur_pos = (size_t)(r.p - f->data);
    int alignment = 32;
    size_t data_offset = ((cur_pos + alignment - 1) / alignment) * alignment;
    f->tensor_data = f->data + data_offset;

    printf("Model: hidden=%d, layers=%d, heads=%d, kv_heads=%d, "
           "inter=%d, vocab=%d, ctx=%d\n",
           f->hidden_size, f->num_layers, f->num_heads, f->num_kv_heads,
           f->intermediate_size, f->vocab_size, f->context_length);

    return 1;
}

gguf_file *gguf_open(const char *path) {
    /* mmap the file */
    uint8_t *data = NULL;
    size_t file_size = 0;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    LARGE_INTEGER sz;
    GetFileSizeEx(hFile, &sz);
    file_size = (size_t)sz.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    data = (uint8_t *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    struct stat st;
    fstat(fd, &st);
    file_size = st.st_size;
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return NULL; }
#endif

    if (!data) { fprintf(stderr, "Failed to map file\n"); return NULL; }

    gguf_file *f = calloc(1, sizeof(gguf_file));
    f->data = data;
    f->file_size = file_size;
    f->owns_data = 1;  /* mmap'd — must unmap on close */
    f->rope_theta = 10000.0f;
    f->logit_scale = 1.0f;

    if (!gguf_parse(f)) {
        free(f);
        return NULL;
    }
    return f;
}

gguf_file *gguf_open_buffer(const uint8_t *data, size_t size) {
    if (!data || size < 8) { fprintf(stderr, "Invalid buffer\n"); return NULL; }

    gguf_file *f = calloc(1, sizeof(gguf_file));
    f->data = (uint8_t *)data;  /* caller owns the buffer */
    f->file_size = size;
    f->owns_data = 0;  /* external buffer — do NOT unmap/free on close */
    f->rope_theta = 10000.0f;
    f->logit_scale = 1.0f;

    if (!gguf_parse(f)) {
        free(f);
        return NULL;
    }
    return f;
}

gguf_tensor_info *gguf_find_tensor(gguf_file *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return &f->tensors[i];
        }
    }
    return NULL;
}

uint8_t *gguf_tensor_data(gguf_file *f, gguf_tensor_info *t) {
    return f->tensor_data + t->offset;
}

void gguf_close(gguf_file *f) {
    if (!f) return;
    if (f->owns_data && f->data) {
#ifdef _WIN32
        UnmapViewOfFile(f->data);
#else
        munmap(f->data, f->file_size);
#endif
    }
    if (f->vocab_tokens) {
        for (int i = 0; i < f->vocab_size; i++) free(f->vocab_tokens[i]);
        free(f->vocab_tokens);
    }
    if (f->vocab_scores) free(f->vocab_scores);
    if (f->merges) {
        for (int i = 0; i < f->n_merges; i++) free(f->merges[i]);
        free(f->merges);
    }
    free(f);
}
