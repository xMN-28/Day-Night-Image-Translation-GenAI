import type { Direction } from "@/lib/models";

export type JobStatus = "queued" | "processing" | "complete" | "failed";

export type StoredJob = {
  id: string;
  status: JobStatus;
  tokenHash: string;
  modelId: string;
  direction: Direction;
  maximumEdge: number;
  inputUrl: string;
  inputPathname: string;
  inputName: string;
  inputWidth?: number;
  inputHeight?: number;
  outputUrl?: string;
  outputPathname?: string;
  metadata?: Record<string, unknown>;
  error?: string;
  createdAt: number;
  updatedAt: number;
  claimedAt?: number;
};

export type WorkerHeartbeat = {
  workerName: string;
  device: string;
  activeJob: string | null;
  availableModels: string[];
  timestamp: number;
};
