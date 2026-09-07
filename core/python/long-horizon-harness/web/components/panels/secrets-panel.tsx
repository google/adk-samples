"use client";
// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


import { memo, useRef, useState, type DragEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { KeyRound, Trash2, Plus, FileUp } from "lucide-react";
import {
  useLhaSecrets,
  putSecret,
  deleteSecret,
  importDotenv,
  type ImportDotenvResult,
} from "@/lib/horizon-secrets";
import { qk } from "@/lib/query-keys";
import { cn } from "@/lib/utils";

interface Staged {
  content: string;
  filename: string;
  preview: ImportDotenvResult;
}

function SecretsPanelImpl() {
  const { data, isLoading } = useLhaSecrets();
  const qc = useQueryClient();
  const invalidateSecrets = () =>
    qc.invalidateQueries({ queryKey: qk.secrets() });
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [staged, setStaged] = useState<Staged | null>(null);
  const [dragging, setDragging] = useState(false);
  const dragDepthRef = useRef(0);

  const secrets = data?.secrets ?? [];
  const showEmpty = !isLoading && secrets.length === 0 && !staged;

  async function onAdd() {
    setErr(null);
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      setErr("Name must be a valid env var (letters, digits, underscore).");
      return;
    }
    setBusy(true);
    try {
      await putSecret(name, value);
      await invalidateSecrets();
      setName("");
      setValue("");
    } catch {
      setErr("Failed to save secret.");
    } finally {
      setBusy(false);
    }
  }

  async function stageFile(file: File) {
    setErr(null);
    setBusy(true);
    try {
      const content = await file.text();
      const preview = await importDotenv(content, false);
      setStaged({ content, filename: file.name, preview });
    } catch {
      setErr("Failed to read or parse file.");
    } finally {
      setBusy(false);
    }
  }

  async function confirmImport() {
    if (!staged) return;
    setBusy(true);
    setErr(null);
    try {
      await importDotenv(staged.content, true);
      await invalidateSecrets();
      setStaged(null);
    } catch {
      setErr("Failed to import secrets.");
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    dragDepthRef.current = 0;
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void stageFile(file);
  }

  return (
    <div className="flex flex-col gap-3 p-2 text-sm">
      {!staged && (
        <label
          className={cn(
            "flex cursor-pointer flex-col items-center gap-1 rounded border border-dashed border-border px-2 py-3 text-xs text-muted-foreground",
            dragging && "border-primary bg-muted",
          )}
          onDragEnter={(e) => {
            e.preventDefault();
            dragDepthRef.current += 1;
            setDragging(true);
          }}
          onDragLeave={() => {
            dragDepthRef.current -= 1;
            if (dragDepthRef.current <= 0) {
              dragDepthRef.current = 0;
              setDragging(false);
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
          }}
          onDrop={onDrop}
        >
          <FileUp className="h-4 w-4" />
          Drop a .env file or click to import
          <input
            type="file"
            className="hidden"
            aria-label="Import .env file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void stageFile(file);
              e.target.value = "";
            }}
          />
        </label>
      )}

      {staged && (
        <div className="flex flex-col gap-2 rounded border border-border p-2 text-xs">
          <p className="font-medium">
            {staged.filename}: {staged.preview.valid.length} key
            {staged.preview.valid.length === 1 ? "" : "s"} found
          </p>
          {staged.preview.valid.length > 0 && (
            <ul className="flex flex-wrap gap-1 font-mono">
              {staged.preview.valid.map((k) => (
                <li key={k} className="rounded bg-muted px-1">
                  {k}
                </li>
              ))}
            </ul>
          )}
          {staged.preview.conflicts.length > 0 && (
            <p className="text-destructive">
              Will overwrite: {staged.preview.conflicts.join(", ")}
            </p>
          )}
          {staged.preview.skipped.length > 0 && (
            <p className="text-muted-foreground">
              Skipped {staged.preview.skipped.length} line
              {staged.preview.skipped.length === 1 ? "" : "s"} (
              {Array.from(
                new Set(staged.preview.skipped.map((s) => s.reason)),
              ).join(", ")}
              )
            </p>
          )}
          <div className="flex gap-2">
            <button
              className={cn(
                "flex-1 rounded bg-primary px-2 py-1 text-primary-foreground",
                (busy || staged.preview.valid.length === 0) && "opacity-50",
              )}
              disabled={busy || staged.preview.valid.length === 0}
              onClick={confirmImport}
              type="button"
            >
              Import {staged.preview.valid.length}
            </button>
            <button
              className="rounded border border-border px-2 py-1"
              onClick={() => setStaged(null)}
              type="button"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <input
          className="rounded border border-border bg-background px-2 py-1"
          placeholder="NAME (e.g. SLACK_API_TOKEN)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          spellCheck={false}
          autoComplete="off"
        />
        <input
          className="rounded border border-border bg-background px-2 py-1"
          placeholder="value"
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoComplete="off"
        />
        <button
          className={cn(
            "flex items-center justify-center gap-1 rounded bg-primary px-2 py-1 text-primary-foreground",
            (busy || !name || !value) && "opacity-50",
          )}
          disabled={busy || !name || !value}
          onClick={onAdd}
          type="button"
        >
          <Plus className="h-4 w-4" /> Add secret
        </button>
        {err && <p className="text-xs text-red-500">{err}</p>}
      </div>

      {showEmpty && (
        <p className="text-xs text-muted-foreground">
          No secrets yet. Add an API key above; the agent can use it in scripts
          without ever seeing the value.
        </p>
      )}

      <ul className="flex flex-col gap-1">
        {secrets.map((s) => (
          <li
            key={s.name}
            className="flex items-center justify-between rounded px-2 py-1 hover:bg-muted"
          >
            <span className="flex items-center gap-2 font-mono text-xs">
              <KeyRound className="h-3.5 w-3.5 text-muted-foreground" />
              {s.name}
              <span className="text-muted-foreground">••••</span>
            </span>
            <button
              aria-label={`Delete ${s.name}`}
              className="text-muted-foreground hover:text-red-500"
              onClick={async () => {
                await deleteSecret(s.name);
                await invalidateSecrets();
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export const SecretsPanel = memo(SecretsPanelImpl);
