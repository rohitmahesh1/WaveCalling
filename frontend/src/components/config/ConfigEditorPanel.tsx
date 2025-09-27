// components/config/ConfigEditorPanel.tsx
import * as React from "react";
import ConfigSections from "@/components/config/ConfigSections";
import ConfigPanel from "./ConfigPanel";
import { CONFIG_SECTIONS, type SectionSpec } from "@/utils/configSchema";
import { useConfigEditor } from "@/hooks/useConfigEditor";
import { useApiBase } from "@/context/ApiContext";

// Lazy-load yaml to avoid bundling if not needed
async function parseYaml(text: string): Promise<any> {
  const mod = await import("yaml");
  return mod.parse(text);
}

// Small nav item shape (no need to import a type from schema)
type NavItem = {
  id: string;
  label: string;
  hint?: string;
  changed?: number;
  errors?: number;
};

function setDeep(obj: any, path: string, value: any) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    if (typeof cur[k] !== "object" || cur[k] === null) cur[k] = {};
    cur = cur[k];
  }
  cur[parts[parts.length - 1]] = value;
}

function deriveDefaultsFromSchema(sections: SectionSpec[]): Record<string, any> {
  const out: Record<string, any> = {};
  for (const s of sections) {
    for (const g of s.groups) {
      for (const f of g.fields) {
        if (typeof f.default !== "undefined") {
          setDeep(out, f.path, f.default);
        }
      }
    }
  }
  return out;
}

function fieldPathsForSection(section: SectionSpec): string[] {
  const paths: string[] = [];
  for (const g of section.groups) {
    for (const f of g.fields) {
      paths.push(f.path);
    }
  }
  return paths;
}

export default function ConfigEditorPanel({
  runId,
  onOverridesChange,
  className,
}: {
  runId?: string | null;
  onOverridesChange?: (json: string) => void;
  className?: string;
}) {
  const apiBase = useApiBase();
  const [baseConfig, setBaseConfig] = React.useState<Record<string, any> | null>(null);
  const [loading, setLoading] = React.useState(false);

  // Load base config.yaml for the selected run if available; otherwise the hook will use schema defaults.
  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!runId) {
        setBaseConfig(null);
        return;
      }
      setLoading(true);
      try {
        const url = `${apiBase}/runs/${encodeURIComponent(runId)}/config.yaml`;
        const r = await fetch(url, { cache: "no-cache" });
        if (!r.ok) throw new Error(`config.yaml not found (${r.status})`);
        const text = await r.text();
        const obj = await parseYaml(text);
        if (!cancelled) setBaseConfig(obj || {});
      } catch {
        if (!cancelled) setBaseConfig(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [apiBase, runId]);

  // IMPORTANT: useConfigEditor accepts { initial, defaults, schema }
  const editor = useConfigEditor({
    initial: baseConfig ?? undefined,
    defaults: deriveDefaultsFromSchema(CONFIG_SECTIONS),
    schema: CONFIG_SECTIONS,
  });

  // Convert overrides object -> pretty JSON and bubble it up to UploadPanel
  const overridesJson = React.useMemo(() => {
    try {
      if (!editor.overrides || Object.keys(editor.overrides).length === 0) return "";
      return JSON.stringify(editor.overrides, null, 2);
    } catch {
      return "";
    }
  }, [editor.overrides]);

  React.useEffect(() => {
    onOverridesChange?.(overridesJson);
  }, [overridesJson, onOverridesChange]);

  // Per-section counts computed locally from editor.changed / editor.errors
  const countsBySection = React.useMemo(() => {
    const changedBySection: Record<string, number> = {};
    const errorsBySection: Record<string, number> = {};
    const changed = editor.changed || {};
    const errors = editor.errors || {};

    for (const s of CONFIG_SECTIONS) {
      const paths = fieldPathsForSection(s);
      let ch = 0;
      let er = 0;
      for (const p of paths) {
        if (changed[p]) ch++;
        if (errors[p]) er++;
      }
      changedBySection[s.id] = ch;
      errorsBySection[s.id] = er;
    }
    return { changedBySection, errorsBySection };
  }, [editor.changed, editor.errors]);

  // Left nav items
  const items: NavItem[] = React.useMemo(() => {
    return CONFIG_SECTIONS.map((s) => ({
      id: s.id,
      label: s.title,
      hint: s.description,
      changed: countsBySection.changedBySection[s.id] || 0,
      errors: countsBySection.errorsBySection[s.id] || 0,
    }));
  }, [countsBySection]);

  const [activeId, setActiveId] = React.useState<string | undefined>(CONFIG_SECTIONS[0]?.id);

  return (
    <div className={className}>
      <ConfigPanel
        title="Run Configuration"
        items={items}
        activeId={activeId}
        onSelect={setActiveId}
        changedCount={editor.changedCount ?? Object.values(editor.changed || {}).filter(Boolean).length}
        errorCount={editor.errorCount ?? Object.values(editor.errors || {}).filter(Boolean).length}
        overridesJson={overridesJson}
        onReset={editor.reset}                           // ⟵ was editor.resetChanges
        onUploadOverrides={(text) => editor.applyOverridesJson?.(text)}
        headerHint={loading ? "Loading base config…" : "Edit settings and export minimal overrides"}
        headerDocsUrl="https://example.com/docs/config"
        renderSection={(_id) => (                        // ⟵ ignore id; ConfigSections renders all cards
          <ConfigSections
            values={editor.value}                        // ⟵ was draft
            onChange={editor.setValue}                   // ⟵ map to onChange
            errors={editor.errors}                       // ⟵ drop 'changed' and 'activeSectionId'
          />
        )}
      />
    </div>
  );
}
