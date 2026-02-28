import axios from "axios";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api/v1";

export interface User {
  id: string;
  username: string;
  email: string;
  avatar?: string;
  provider: "local" | "google";
}

export interface AuthResponse {
  success: boolean;
  message?: string;
  error?: string;
  data?: {
    user: User;
    token: string;
  };
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

export interface LoginData {
  email: string;
  password: string;
}

const TOKEN_KEY = "safesight_token";
const USER_KEY = "safesight_user";

class AuthService {
  private token: string | null = null;

  constructor() {
    if (typeof window !== "undefined") {
      this.token = localStorage.getItem(TOKEN_KEY);
    }
  }

  getToken(): string | null {
    if (typeof window !== "undefined") {
      return localStorage.getItem(TOKEN_KEY);
    }
    return this.token;
  }

  getUser(): User | null {
    if (typeof window !== "undefined") {
      const userStr = localStorage.getItem(USER_KEY);
      if (userStr) {
        try {
          return JSON.parse(userStr);
        } catch {
          return null;
        }
      }
    }
    return null;
  }

  setAuth(token: string, user: User): void {
    this.token = token;
    if (typeof window !== "undefined") {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
  }

  clearAuth(): void {
    this.token = null;
    if (typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }

  async register(data: RegisterData): Promise<AuthResponse> {
    try {
      const response = await axios.post<AuthResponse>(
        `${API_BASE_URL}/auth/register`,
        data,
      );

      if (response.data.success && response.data.data) {
        this.setAuth(response.data.data.token, response.data.data.user);
      }

      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || "Registration failed",
      };
    }
  }

  async login(data: LoginData): Promise<AuthResponse> {
    try {
      const response = await axios.post<AuthResponse>(
        `${API_BASE_URL}/auth/login`,
        data,
      );

      if (response.data.success && response.data.data) {
        this.setAuth(response.data.data.token, response.data.data.user);
      }

      return response.data;
    } catch (error: any) {
      return {
        success: false,
        error: error.response?.data?.error || "Login failed",
      };
    }
  }

  async getCurrentUser(): Promise<AuthResponse> {
    try {
      const token = this.getToken();
      if (!token) {
        return { success: false, error: "Not authenticated" };
      }

      const response = await axios.get<AuthResponse>(
        `${API_BASE_URL}/auth/me`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );

      return response.data;
    } catch (error: any) {
      if (error.response?.status === 401) {
        this.clearAuth();
      }
      return {
        success: false,
        error: error.response?.data?.error || "Failed to get user",
      };
    }
  }

  async logout(): Promise<void> {
    try {
      const token = this.getToken();
      if (token) {
        await axios.post(
          `${API_BASE_URL}/auth/logout`,
          {},
          { headers: { Authorization: `Bearer ${token}` } },
        );
      }
    } finally {
      this.clearAuth();
    }
  }

  getGoogleAuthUrl(): string {
    return `${API_BASE_URL}/auth/google`;
  }
}

export const authService = new AuthService();
export default authService;
