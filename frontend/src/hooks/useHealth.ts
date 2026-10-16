import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth } from "../services/system";
import type { HealthResponse } from "../types/system";

const POLL_INTERVAL_MS = 15000;

export function useHealth(poll = false) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const res = await getHealth();
      setHealth(res);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    if (poll) {
      const loop = () => {
        timerRef.current = setTimeout(async () => {
          await checkHealth();
          loop();
        }, POLL_INTERVAL_MS);
      };
      loop();
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [poll]);

  return { health, loading, error, checkHealth };
}
