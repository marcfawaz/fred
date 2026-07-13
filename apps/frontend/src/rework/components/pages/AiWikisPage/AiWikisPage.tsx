// Copyright Thales 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { Box } from "@mui/material";
import { useCallback, useContext, useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useLocation, useParams } from "react-router-dom";
import { ApplicationContext } from "../../../../app/ApplicationContextProvider";
import type { ThemeMode } from "../../../../app/ApplicationContextStruct";
import { getProperty } from "../../../../common/config";

function normalizeFrontendUrl(url: string): string {
  if (!url || url === "/") {
    return "/";
  }
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

function encodePathSegments(path: string | undefined): string {
  if (!path) {
    return "";
  }
  const encoded = path
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return encoded ? `/${encoded}` : "";
}

export type FredThemeMessage = {
  type: "fred:theme";
  themeMode: ThemeMode;
  effectiveTheme: "light" | "dark";
};

export type FredLanguage = "en" | "fr";

export type FredLanguageMessage = {
  type: "fred:language";
  language: FredLanguage;
};

export function normalizeFredLanguage(language: string | undefined): FredLanguage {
  return language?.toLowerCase().startsWith("fr") ? "fr" : "en";
}

export function buildFredThemeMessage(themeMode: ThemeMode, darkMode: boolean): FredThemeMessage {
  return {
    type: "fred:theme",
    themeMode,
    effectiveTheme: darkMode ? "dark" : "light",
  };
}

export function buildFredLanguageMessage(language: FredLanguage): FredLanguageMessage {
  return {
    type: "fred:language",
    language,
  };
}

export function getAiWikisTargetOrigin(aiWikisFrontendUrl: string, currentOrigin: string): string {
  if (!aiWikisFrontendUrl || aiWikisFrontendUrl.startsWith("/")) {
    return currentOrigin;
  }
  return new URL(aiWikisFrontendUrl).origin;
}

export function buildAiWikisIframeSrc({
  aiWikisFrontendUrl,
  teamId,
  splatPath,
  search,
  themeMode,
  darkMode,
  language,
}: {
  aiWikisFrontendUrl: string;
  teamId: string;
  splatPath?: string;
  search?: string;
  themeMode: ThemeMode;
  darkMode: boolean;
  language: FredLanguage;
}) {
  const baseUrl = normalizeFrontendUrl(aiWikisFrontendUrl || "/ai-wikis");
  const encodedTeamId = encodeURIComponent(teamId);
  const encodedSplatPath = encodePathSegments(splatPath);
  const params = new URLSearchParams(search ?? "");
  params.set("theme", darkMode ? "dark" : "light");
  params.set("themeMode", themeMode);
  params.set("lng", language);
  const normalizedSearch = params.size > 0 ? `?${params.toString()}` : "";
  if (baseUrl === "/") {
    return `/embed/team/${encodedTeamId}${encodedSplatPath}${normalizedSearch}`;
  }
  return `${baseUrl}/embed/team/${encodedTeamId}${encodedSplatPath}${normalizedSearch}`;
}

export default function AiWikisPage() {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const location = useLocation();
  const { teamId = "personal", "*": splatPath } = useParams<{ teamId: string; "*": string }>();
  const { themeMode, darkMode } = useContext(ApplicationContext);
  const { i18n } = useTranslation();
  const aiWikisFrontendUrl = getProperty("aiWikisFrontendUrl") || "/ai-wikis";
  const language = useMemo(() => normalizeFredLanguage(i18n.language), [i18n.language]);
  const themeMessage = useMemo(() => buildFredThemeMessage(themeMode, darkMode), [themeMode, darkMode]);
  const languageMessage = useMemo(() => buildFredLanguageMessage(language), [language]);
  const iframeSrc = useMemo(
    () =>
      buildAiWikisIframeSrc({
        aiWikisFrontendUrl,
        teamId,
        splatPath,
        search: location.search,
        themeMode,
        darkMode,
        language,
      }),
    [aiWikisFrontendUrl, darkMode, language, location.search, splatPath, teamId, themeMode],
  );
  const targetOrigin = useMemo(
    () => getAiWikisTargetOrigin(aiWikisFrontendUrl, globalThis.location?.origin ?? "http://localhost"),
    [aiWikisFrontendUrl],
  );

  const postThemeMessage = useCallback(() => {
    iframeRef.current?.contentWindow?.postMessage(themeMessage, targetOrigin);
  }, [targetOrigin, themeMessage]);

  const postLanguageMessage = useCallback(() => {
    iframeRef.current?.contentWindow?.postMessage(languageMessage, targetOrigin);
  }, [languageMessage, targetOrigin]);

  useEffect(() => {
    postThemeMessage();
    postLanguageMessage();
  }, [postLanguageMessage, postThemeMessage]);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "stretch",
        flex: 1,
        overflow: "hidden",
        minHeight: 0,
      }}
    >
      <Box
        component="iframe"
        ref={iframeRef}
        title="AI Wikis"
        src={iframeSrc}
        onLoad={() => {
          postThemeMessage();
          postLanguageMessage();
        }}
        sx={{
          width: "100%",
          height: "100%",
          border: 0,
          flex: 1,
          minHeight: 0,
        }}
      />
    </Box>
  );
}
