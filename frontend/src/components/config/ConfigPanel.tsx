import * as React from "react";
import ConfigHeader from "@/components/config/ConfigHeader";
import ConfigCategoryNav, {
  type ConfigNavItem,
} from "@/components/config/ConfigCategoryNav";

type Props = {
  /** Items for the left-side category navigation. */
  items: ConfigNavItem[];
  /** The currently active section id. */
  activeId: string | undefined;
  /** Called when a new section is selected. */
  onSelect: (id: string) => void;

  /** Renders the content for the active section. */
  renderSection: (id: string | undefined) => React.ReactNode;

  /** Counts for header badges. */
  changedCount?: number;
  errorCount?: number;

  /** Current overrides JSON text to enable Copy/Download actions. */
  overridesJson?: string;

  /** Handlers for header actions. */
  onReset?: () => void;
  onUploadOverrides?: (text: string) => void;
  onCopyOverrides?: () => void;

  /** Optional header extras (e.g., “Apply”, “Save”, etc.). */
  headerRightExtra?: React.ReactNode;

  /** Optional header hint/docs. */
  headerHint?: string;
  headerDocsUrl?: string;

  /** Panel-level styling. */
  className?: string;
  headerClassName?: string;
  navClassName?: string;
  contentClassName?: string;

  /** Title for the header. */
  title?: string;
};

export default function ConfigPanel({
  items,
  activeId,
  onSelect,
  renderSection,

  changedCount = 0,
  errorCount = 0,

  overridesJson,
  onReset,
  onUploadOverrides,
  onCopyOverrides,

  headerRightExtra,
  headerHint,
  headerDocsUrl,

  className,
  headerClassName,
  navClassName,
  contentClassName,

  title = "Configuration",
}: Props) {
  return (
    <section className={`flex flex-col gap-3 ${className || ""}`}>
      <ConfigHeader
        title={title}
        changedCount={changedCount}
        errorCount={errorCount}
        overridesJson={overridesJson}
        onReset={onReset}
        onUploadOverrides={onUploadOverrides}
        onCopyOverrides={onCopyOverrides}
        rightExtra={headerRightExtra}
        className={headerClassName}
        hint={headerHint}
        docsUrl={headerDocsUrl}
      />

      <div className="grid grid-cols-1 lg:grid-cols-[280px,1fr] gap-4">
        {/* <ConfigCategoryNav
          items={items}
          activeId={activeId}
          onSelect={onSelect}
          className={navClassName}
        /> */}

        <div
          className={`rounded-xl border border-slate-700/50 bg-console-700 p-4 min-h-[300px] ${contentClassName || ""}`}
        >
          <div className="pr-1">
            {renderSection(activeId)}
          </div>
        </div>
      </div>
    </section>
  );
}
