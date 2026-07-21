import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApplicationContext } from "../../../../app/ApplicationContextProvider";
import AiWikisPage, {
  buildAiWikisAuthVersion,
  buildAiWikisIframeSrc,
  buildFredLanguageMessage,
  buildFredThemeMessage,
  getAiWikisTargetOrigin,
  normalizeFredLanguage,
  resolveAiWikisFrontendUrl,
  validateAiWikisFrontendSameOrigin,
} from "./AiWikisPage";

let currentLanguage = "en";
let aiWikisFrontendUrl = "/ai-wikis";
let canReadWikis = true;
let selectedTeam: { id: string; permissions?: string[] } | undefined = {
  id: "team-42",
  permissions: ["can_read_wikis"],
};
let isPersonalTeam = false;

vi.mock("react-i18next", () => ({
  useTranslation: vi.fn(() => ({
    i18n: {
      language: currentLanguage,
    },
  })),
}));

vi.mock("../../../../common/config", () => ({
  getProperty: vi.fn((key: string) => {
    if (key === "aiWikisFrontendUrl") {
      return aiWikisFrontendUrl;
    }
    return "";
  }),
}));

vi.mock("../../../../hooks/useSelectedTeam", () => ({
  useSelectedTeam: vi.fn(() => ({
    selectedTeam,
    isPersonalTeam,
  })),
}));

vi.mock("../../../core/hooks/useTeamCapabilities", () => ({
  useTeamCapabilities: vi.fn(() => ({
    canReadWikis,
  })),
}));

vi.mock("../../../../security/KeycloakService", () => ({
  KeyCloakService: {
    GetTokenParsed: vi.fn(() => ({
      sub: "priya-subject",
      preferred_username: "priya",
      iat: 123,
      sid: "session-a",
    })),
    GetUserId: vi.fn(() => "priya-subject"),
  },
}));

function renderAt(path: string) {
  return renderToStaticMarkup(
    <ApplicationContext.Provider
      value={{
        themeMode: "dark",
        darkMode: true,
        isSidebarCollapsed: false,
        toggleSidebar: vi.fn(),
        setThemeMode: vi.fn(),
      }}
    >
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/team/:teamId/wikis/*" element={<AiWikisPage />} />
        </Routes>
      </MemoryRouter>
    </ApplicationContext.Provider>,
  );
}

describe("AiWikisPage", () => {
  beforeEach(() => {
    currentLanguage = "en";
    aiWikisFrontendUrl = "/ai-wikis";
    canReadWikis = true;
    selectedTeam = { id: "team-42", permissions: ["can_read_wikis"] };
    isPersonalTeam = false;
  });

  it("normalizes Fred language values", () => {
    expect(normalizeFredLanguage("fr-FR")).toBe("fr");
    expect(normalizeFredLanguage("en-US")).toBe("en");
    expect(normalizeFredLanguage(undefined)).toBe("en");
  });

  it("renders an iframe for the team root route", () => {
    const html = renderAt("/team/team-42/wikis");
    const authVersion = buildAiWikisAuthVersion(
      { sub: "priya-subject", preferred_username: "priya", iat: 123, sid: "session-a" },
      "priya-subject",
    );
    expect(html).toContain("<iframe");
    expect(html).toContain('title="AI Wikis"');
    expect(html).toContain(`src="/ai-wikis/embed/team/team-42?theme=dark&amp;themeMode=dark&amp;lng=en&amp;authv=${authVersion}"`);
  });

  it("does not render the iframe for a collaborative team without canReadWikis", () => {
    canReadWikis = false;
    selectedTeam = { id: "team-42", permissions: [] };
    const html = renderAt("/team/team-42/wikis");

    expect(html).not.toContain("<iframe");
    expect(html).toContain("AI Wikis are not available for this team.");
  });

  it("fails closed while collaborative team permissions are unresolved", () => {
    canReadWikis = false;
    selectedTeam = undefined;
    const html = renderAt("/team/team-42/wikis");

    expect(html).not.toContain("<iframe");
    expect(html).toContain("AI Wikis are not available for this team.");
  });

  it("preserves personal AI Wikis access without collaborative canReadWikis", () => {
    canReadWikis = false;
    selectedTeam = undefined;
    isPersonalTeam = true;
    const html = renderAt("/team/personal/wikis");

    expect(html).toContain("<iframe");
    expect(html).toContain('src="/ai-wikis/embed/team/personal?theme=dark&amp;themeMode=dark&amp;lng=en&amp;authv=');
  });

  it("preserves the nested wiki subpath and query string", () => {
    currentLanguage = "fr";
    const html = renderAt("/team/team-42/wikis/wiki-123/pages/page-456?view=compact");
    expect(html).toContain(
      'src="/ai-wikis/embed/team/team-42/wiki-123/pages/page-456?view=compact&amp;theme=dark&amp;themeMode=dark&amp;lng=fr&amp;authv=',
    );
  });

  it("encodes team and path segments safely", () => {
    currentLanguage = "en";
    const html = renderAt("/team/team space/wikis/wiki folder/pages/page name");
    expect(html).toContain(
      'src="/ai-wikis/embed/team/team%20space/wiki%20folder/pages/page%20name?theme=dark&amp;themeMode=dark&amp;lng=en&amp;authv=',
    );
  });

  it("builds iframe src with preserved query params, theme params, and language", () => {
    expect(
      buildAiWikisIframeSrc({
        aiWikisFrontendUrl: "/ai-wikis",
        teamId: "team-42",
        splatPath: "wiki-123/pages/page-456",
        search: "?view=compact",
        themeMode: "system",
        darkMode: true,
        language: "fr",
        authVersion: "auth-1",
      }),
    ).toBe("/ai-wikis/embed/team/team-42/wiki-123/pages/page-456?view=compact&theme=dark&themeMode=system&lng=fr&authv=auth-1");
  });

  it("builds a non-secret auth version from identity and session claims", () => {
    const first = buildAiWikisAuthVersion({ sub: "user-a", preferred_username: "priya", iat: 100, sid: "session-a" }, "fallback");
    const second = buildAiWikisAuthVersion({ sub: "user-b", preferred_username: "elena", iat: 101, sid: "session-b" }, "fallback");

    expect(first).not.toBe(second);
    expect(first).not.toContain("user-a");
    expect(second).not.toContain("elena");
  });

  it("returns same-origin target for relative iframe urls", () => {
    expect(getAiWikisTargetOrigin("/ai-wikis", "http://localhost:5173")).toBe("http://localhost:5173");
  });

  it("does not use an unsafe target origin for rejected absolute iframe urls", () => {
    expect(getAiWikisTargetOrigin("https://wiki.example.test/ai-wikis", "http://localhost:5173")).toBe(
      "http://localhost:5173",
    );
  });

  it("validates aiWikisFrontendUrl same-origin compatibility", () => {
    expect(validateAiWikisFrontendSameOrigin("/ai-wikis", "https://fred.example")).toBeNull();
    expect(validateAiWikisFrontendSameOrigin("https://fred.example/ai-wikis", "https://fred.example")).toBeNull();
    expect(validateAiWikisFrontendSameOrigin("//wiki.example/ai-wikis", "https://fred.example")).toContain(
      "same public origin",
    );
    expect(validateAiWikisFrontendSameOrigin("https://wiki.example/ai-wikis", "https://fred.example")).toContain(
      "same public origin",
    );
    expect(validateAiWikisFrontendSameOrigin("not-a-url", "https://fred.example")).toContain("same public origin");
  });

  it("centralizes aiWikisFrontendUrl parsing behavior", () => {
    expect(resolveAiWikisFrontendUrl("/ai-wikis", "https://fred.example")).toMatchObject({
      valid: true,
      frontendUrl: "/ai-wikis",
      targetOrigin: "https://fred.example",
    });
    expect(resolveAiWikisFrontendUrl("https://fred.example/ai-wikis", "https://fred.example")).toMatchObject({
      valid: true,
      frontendUrl: "https://fred.example/ai-wikis",
      targetOrigin: "https://fred.example",
    });
    expect(resolveAiWikisFrontendUrl("//wiki.example/ai-wikis", "https://fred.example")).toMatchObject({
      valid: false,
      frontendUrl: "",
      targetOrigin: "https://fred.example",
    });
  });

  it("renders a configuration error and no iframe for invalid frontend URL values", () => {
    aiWikisFrontendUrl = "//wiki.example/ai-wikis";
    const html = renderAt("/team/team-42/wikis");

    expect(html).not.toContain("<iframe");
    expect(html).toContain("same public origin");
  });

  it("builds the fred theme message payload", () => {
    expect(buildFredThemeMessage("system", true)).toEqual({
      type: "fred:theme",
      themeMode: "system",
      effectiveTheme: "dark",
    });
  });

  it("builds the fred language message payload", () => {
    expect(buildFredLanguageMessage("fr")).toEqual({
      type: "fred:language",
      language: "fr",
    });
  });
});
