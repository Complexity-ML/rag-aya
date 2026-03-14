/**
 * HTTP server helpers implementation.
 * INL 2025
 */
#include "http.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void send_response(SOCKET sock, int status, const char *status_text,
                   const char *content_type, const char *body) {
    char header[512];
    int body_len = (int)strlen(body);
    snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    send(sock, header, (int)strlen(header), 0);
    send(sock, body, body_len, 0);
}

void send_sse_start(SOCKET sock) {
    const char *header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";
    send(sock, header, (int)strlen(header), 0);
}

void send_sse_token(SOCKET sock, const char *token) {
    char buf[512];
    char escaped[256];
    int j = 0;
    for (int i = 0; token[i] && j < 250; i++) {
        if (token[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
        else if (token[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
        else if (token[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
        else escaped[j++] = token[i];
    }
    escaped[j] = '\0';
    snprintf(buf, sizeof(buf), "data: {\"token\":\"%s\"}\n\n", escaped);
    send(sock, buf, (int)strlen(buf), 0);
}

void send_sse_done(SOCKET sock) {
    const char *msg = "data: [DONE]\n\n";
    send(sock, msg, (int)strlen(msg), 0);
}

/* ---- JSON helpers ---- */

const char *json_get_string(const char *json, const char *key,
                            char *buf, int buf_sz) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < buf_sz - 1) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            if (*p == 'n') buf[i++] = '\n';
            else if (*p == 't') buf[i++] = '\t';
            else if (*p == '"') buf[i++] = '"';
            else if (*p == '\\') buf[i++] = '\\';
            else buf[i++] = *p;
        } else {
            buf[i++] = *p;
        }
        p++;
    }
    buf[i] = '\0';
    return buf;
}

int json_get_int(const char *json, const char *key, int default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return atoi(p);
}

float json_get_float(const char *json, const char *key, float default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return (float)atof(p);
}
