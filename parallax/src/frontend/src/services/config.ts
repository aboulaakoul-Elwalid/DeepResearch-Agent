const STORAGE_KEY = 'parallax.apiBaseUrl';

const getDefaultBaseUrl = () =>
  (import.meta.env.VITE_API_BASE_URL as string | undefined)
  || (import.meta.env.DEV ? '/proxy-api' : '');

export const getApiBaseUrl = (): string => {
  try {
    if (typeof localStorage !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) return stored;
    }
  } catch (_err) {
    // ignore storage errors
  }
  return getDefaultBaseUrl();
};

export const setApiBaseUrl = (url: string) => {
  try {
    if (typeof localStorage !== 'undefined') {
      if (url) {
        localStorage.setItem(STORAGE_KEY, url);
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  } catch (_err) {
    // ignore storage errors
  }
};

export const resetApiBaseUrl = () => {
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch (_err) {
    // ignore
  }
};
