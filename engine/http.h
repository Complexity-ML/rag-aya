/**
 * Minimal HTTP server helpers + JSON parsing.
 * INL 2025
 */
#ifndef HTTP_H
#define HTTP_H

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef int socklen_t;
  #define CLOSE_SOCKET closesocket
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #define CLOSE_SOCKET close
  typedef int SOCKET;
  #define INVALID_SOCKET -1
#endif

/* Send a complete HTTP response */
void send_response(SOCKET sock, int status, const char *status_text,
                   const char *content_type, const char *body);

/* SSE (Server-Sent Events) streaming */
void send_sse_start(SOCKET sock);
void send_sse_token(SOCKET sock, const char *token);
void send_sse_done(SOCKET sock);

/* Minimal JSON parsing from string body */
const char *json_get_string(const char *json, const char *key,
                            char *buf, int buf_sz);
int   json_get_int(const char *json, const char *key, int default_val);
float json_get_float(const char *json, const char *key, float default_val);

#endif /* HTTP_H */
