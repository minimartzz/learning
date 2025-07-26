import React from "react";
import Ping from "./Ping";
import { client } from "@/sanity/lib/client";
import { STARTUP_VIEWS_QUERY } from "@/lib/queries";
import { writeClient } from "@/sanity/lib/write";
import { after } from "next/server";

const View = async ({ id }: { id: string }) => {
  const { views: totalViews } = await client
    .withConfig({ useCdn: false })
    .fetch(STARTUP_VIEWS_QUERY, { id });

  await after(() =>
    writeClient
      .patch(id)
      .set({ views: totalViews + 1 })
      .commit()
  );

  return (
    <>
      <div className="view-container">
        <div className="absolute -top-2 -right-2">
          <Ping />
        </div>

        <p className="view-text">
          <span className="font-black">
            {totalViews} View{totalViews == 1 ? "" : "s"}
          </span>
        </p>
      </div>
    </>
  );
};

export default View;
