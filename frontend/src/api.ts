export interface SignalPlot {
  channels: string[];
  time: number[];
  series: { channel: string; values: number[] }[];
  flat: number[];
}

export interface ApplianceCommand {
  stimulus_id: number;
  label: string;
}

export interface ApplianceInfo {
  name: string;
  class_id: number;
  folder: string;
  subjects: number;
  paradigm: string;
  environment: string;
  stimuli: number;
  target_ratio: string;
  repetitions_per_block: number;
  commands: ApplianceCommand[];
  note: string;
}

export async function fetchAppliances(): Promise<ApplianceInfo[]> {
  const res = await fetch('/api/appliances', { cache: 'no-store' });
  if (!res.ok) throw new Error(`Catalog request failed (${res.status})`);
  const j = await res.json();
  return j.appliances as ApplianceInfo[];
}

export interface PredictedCommand {
  stimulus_id: number;
  label: string;
  confidence: number;
  command_scores: Record<string, number>;
  erp_scores: Record<string, number>;
  repetitions: Record<string, number>;
  method: string;
  file: string;
  context_appliance: string;
}
export interface PredictResponse {
  predicted_appliance: string;
  class_id: number;
  confidence_scores: Record<string, number>;
  p300_detected: boolean;
  p300_score?: number;
  signal_preview: number[];
  signal_plot?: SignalPlot;
  meta?: Record<string, any>;
  file?: string;
  files_processed?: number;
  files?: string[];
  used_fallback?: boolean;
  appliance_info?: ApplianceInfo;
  predicted_command?: PredictedCommand | null;
  model_source?: string;
}

export async function predictFile(file: File, aggregate = 'first', appliance = ''): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append('file', file);
  const q = new URLSearchParams({ aggregate, appliance });
  const res = await fetch(`/api/predict?${q.toString()}`, { method: 'POST', body: fd });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `Request failed (${res.status})`);
  }
  return res.json();
}
