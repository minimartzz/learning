import "server-only";

import { defineLive } from "next-sanity";
import { client } from "@/sanity/lib/client";

export const { sanityFetch, SanityLive } = defineLive({
    client,
    serverToken: false,
    browserToken: false,
});
