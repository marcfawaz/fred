import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { ApplicationContext } from "../../../../app/ApplicationContextProvider";
import AiWikisPage, {
  buildAiWikisIframeSrc,
  buildFredLanguageMessage,
  buildFredThemeMessage,
  getAiWikisTargetOrigin,
  normalizeFredLanguage,
} from "./AiWikisPage";

let currentLanguage = "en";

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
      return "/ai-wikis";
    }
    return "";
  }),
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
  it("normalizes Fred language values", () => {
    expect(normalizeFredLanguage("fr-FR")).toBe("fr");
    expect(normalizeFredLanguage("en-US")).toBe("en");
    expect(normalizeFredLanguage(undefined)).toBe("en");
  });

  it("renders an iframe for the team root route", () => {
    currentLanguage = "en";
    const html = renderAt("/team/team-42/wikis");
    expect(html).toContain("<iframe");
    expect(html).toContain('title="AI Wikis"');
    expect(html).toContain('src="/ai-wikis/embed/team/team-42?theme=dark&amp;themeMode=dark&amp;lng=en"');
  });

  it("preserves the nested wiki subpath and query string", () => {
    currentLanguage = "fr";
    const html = renderAt("/team/team-42/wikis/wiki-123/pages/page-456?view=compact");
    expect(html).toContain(
      'src="/ai-wikis/embed/team/team-42/wiki-123/pages/page-456?view=compact&amp;theme=dark&amp;themeMode=dark&amp;lng=fr"',
    );
  });

  it("encodes team and path segments safely", () => {
    currentLanguage = "en";
    const html = renderAt("/team/team space/wikis/wiki folder/pages/page name");
    expect(html).toContain(
      'src="/ai-wikis/embed/team/team%20space/wiki%20folder/pages/page%20name?theme=dark&amp;themeMode=dark&amp;lng=en"',
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
      }),
    ).toBe("/ai-wikis/embed/team/team-42/wiki-123/pages/page-456?view=compact&theme=dark&themeMode=system&lng=fr");
  });

  it("returns same-origin target for relative iframe urls", () => {
    expect(getAiWikisTargetOrigin("/ai-wikis", "http://localhost:5173")).toBe("http://localhost:5173");
  });

  it("returns external origin for absolute iframe urls", () => {
    expect(getAiWikisTargetOrigin("https://wiki.example.test/ai-wikis", "http://localhost:5173")).toBe(
      "https://wiki.example.test",
    );
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
