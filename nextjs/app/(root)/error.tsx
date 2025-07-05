// app/global-error.tsx
"use client"; // Error boundaries must be Client Components

export default function Error({
  error,
}: {
  error: Error & { digest?: string };
}) {
  return (
    // global-error must include html and body tags
    <div>
      <h2>⚠️⚠️ ROOT ERROR! ⚠️⚠️</h2>
    </div>
  );
}
