import React from "react";

const layout = ({ children }: { children: React.ReactNode }) => {
  return (
    <html>
      <body>
        <h1 className="text-3xl">Dashboard</h1>
        {children}
      </body>
    </html>
  );
};

export default layout;
