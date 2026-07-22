/**
 * Trả về đường dẫn gốc (Base URL) khi gọi API sang Backend.
 * 
 * - Nếu cấu hình biến NEXT_PUBLIC_API_URL (ví dụ khi gọi trực tiếp sang domain Ngrok riêng của Backend): trả về URL đó.
 * - Mặc định (chuỗi rỗng ''): các request như `/api/attendance` hoặc `/static/anchors/...` sẽ được gửi về chính domain hiện tại
 *   (bất kể là http://localhost:3000, http://192.168.1.x:3000 hay https://xxx.ngrok-free.app).
 *   Next.js server thông qua cấu hình `rewrites` trong `next.config.ts` sẽ tự động chuyển tiếp (proxy) xuống Backend FastAPI
 *   chạy ở `http://127.0.0.1:8000`. Điều này giúp khắc phục giới hạn 1 tunnel của Ngrok miễn phí và tránh lỗi CORS.
 */
export function getApiBaseUrl(): string {
  if (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, '');
  }
  return '';
}
