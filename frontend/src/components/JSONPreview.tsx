export default function JSONPreview({ data }: { data: unknown }) {
  return (
    <pre style={{ whiteSpace: "pre-wrap", background: "#0b0e14", padding: 12, borderRadius: 8 }}>
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
