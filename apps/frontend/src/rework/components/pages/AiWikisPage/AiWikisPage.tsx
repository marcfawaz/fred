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
import { useSelectedTeam } from "../../../../hooks/useSelectedTeam";
import { useTeamCapabilities } from "../../../core/hooks/useTeamCapabilities";
import { isPersonalTeamId } from "@shared/utils/teamId";
import { KeyCloakService } from "../../../../security/KeycloakService";

function normalizeFrontendUrl(url: string): string {
  if (!url || url === "/") {
    return "/";
  }
  return url.endsWith("/") ? url.slice(0, -1) : url;
}

export type AiWikisFrontendUrlResolution =
  | { valid: true; frontendUrl: string; targetOrigin: string; error?: undefined }
  | { valid: false; frontendUrl: ""; targetOrigin: string; error: string };

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

function stableIdentityHash(value: string): string {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) + hash) ^ value.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

export function buildAiWikisAuthVersion(tokenPayload: Record<string, unknown> | null | undefined, fallbackUserId?: string | null): string {
  const subject = typeof tokenPayload?.sub === "string" && tokenPayload.sub ? tokenPayload.sub : fallbackUserId ?? "anonymous";
  const username = typeof tokenPayload?.preferred_username === "string" ? tokenPayload.preferred_username : "";
  const issuedAt = typeof tokenPayload?.iat === "number" || typeof tokenPayload?.iat === "string" ? String(tokenPayload.iat) : "";
  const sessionId = typeof tokenPayload?.sid === "string"
    ? tokenPayload.sid
    : typeof tokenPayload?.session_state === "string"
      ? tokenPayload.session_state
      : "";
  return stableIdentityHash([subject, username, issuedAt, sessionId].join(":"));
}

export function getAiWikisTargetOrigin(aiWikisFrontendUrl: string, currentOrigin: string): string {
  return resolveAiWikisFrontendUrl(aiWikisFrontendUrl, currentOrigin).targetOrigin;
}

export function resolveAiWikisFrontendUrl(
  aiWikisFrontendUrl: string,
  currentOrigin: string,
): AiWikisFrontendUrlResolution {
  const value = aiWikisFrontendUrl || "/ai-wikis";
  const error =
    "AI Wikis must be exposed through the same public origin as Fred when using the fred-local-storage token bridge.";
  if (value.startsWith("//")) {
    return { valid: false, frontendUrl: "", targetOrigin: currentOrigin, error };
  }
  if (value.startsWith("/")) {
    return { valid: true, frontendUrl: normalizeFrontendUrl(value), targetOrigin: currentOrigin };
  }
  try {
    const target = new URL(value);
    if (target.origin !== currentOrigin) {
      return { valid: false, frontendUrl: "", targetOrigin: currentOrigin, error };
    }
    return { valid: true, frontendUrl: normalizeFrontendUrl(target.toString()), targetOrigin: target.origin };
  } catch {
    return { valid: false, frontendUrl: "", targetOrigin: currentOrigin, error };
  }
}

export function validateAiWikisFrontendSameOrigin(aiWikisFrontendUrl: string, currentOrigin: string): string | null {
  const resolved = resolveAiWikisFrontendUrl(aiWikisFrontendUrl, currentOrigin);
  return resolved.valid ? null : resolved.error;
}

export function buildAiWikisIframeSrc({
  aiWikisFrontendUrl,
  teamId,
  splatPath,
  search,
  themeMode,
  darkMode,
  language,
  authVersion,
}: {
  aiWikisFrontendUrl: string;
  teamId: string;
  splatPath?: string;
  search?: string;
  themeMode: ThemeMode;
  darkMode: boolean;
  language: FredLanguage;
  authVersion?: string;
}) {
  const resolved = resolveAiWikisFrontendUrl(
    aiWikisFrontendUrl || "/ai-wikis",
    globalThis.location?.origin ?? "http://localhost",
  );
  if (!resolved.valid) {
    return "";
  }
  const baseUrl = resolved.frontendUrl;
  const encodedTeamId = encodeURIComponent(teamId);
  const encodedSplatPath = encodePathSegments(splatPath);
  const params = new URLSearchParams(search ?? "");
  params.set("theme", darkMode ? "dark" : "light");
  params.set("themeMode", themeMode);
  params.set("lng", language);
  if (authVersion) {
    params.set("authv", authVersion);
  }
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
  const { selectedTeam, isPersonalTeam } = useSelectedTeam();
  const { canReadWikis } = useTeamCapabilities(selectedTeam);
  const isPersonalAiWikisRoute = isPersonalTeam || teamId === "personal" || isPersonalTeamId(teamId);
  const currentOrigin = globalThis.location?.origin ?? "http://localhost";
  const resolvedAiWikisFrontendUrl = useMemo(
    () => resolveAiWikisFrontendUrl(aiWikisFrontendUrl, currentOrigin),
    [aiWikisFrontendUrl, currentOrigin],
  );
  const sameOriginConfigurationError = resolvedAiWikisFrontendUrl.valid ? null : resolvedAiWikisFrontendUrl.error;
  const language = useMemo(() => normalizeFredLanguage(i18n.language), [i18n.language]);
  const authVersion = buildAiWikisAuthVersion(KeyCloakService.GetTokenParsed?.(), KeyCloakService.GetUserId?.());
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
        authVersion,
      }),
    [aiWikisFrontendUrl, authVersion, darkMode, language, location.search, splatPath, teamId, themeMode],
  );
  const targetOrigin = resolvedAiWikisFrontendUrl.targetOrigin;

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

  if (sameOriginConfigurationError) {
    return (
      <Box role="alert" sx={{ p: 3 }}>
        {sameOriginConfigurationError}
      </Box>
    );
  }

  if (!isPersonalAiWikisRoute && !canReadWikis) {
    return (
      <Box role="alert" sx={{ p: 3 }}>
        AI Wikis are not available for this team.
      </Box>
    );
  }

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
        key={authVersion}
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
