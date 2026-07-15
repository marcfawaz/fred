import { describe, expect, it } from "vitest";
import { shouldShowAiWikisNavigation } from "./TeamContentNavbar";

describe("shouldShowAiWikisNavigation", () => {
  it("shows collaborative AI Wikis only when the feature and canReadWikis are both enabled", () => {
    expect(shouldShowAiWikisNavigation({ isAiWikisEnabled: true, isPersonalTeam: false, canReadWikis: true })).toBe(
      true,
    );
    expect(shouldShowAiWikisNavigation({ isAiWikisEnabled: true, isPersonalTeam: false, canReadWikis: false })).toBe(
      false,
    );
  });

  it("fails closed while permissions are unresolved and preserves personal access", () => {
    expect(shouldShowAiWikisNavigation({ isAiWikisEnabled: true, isPersonalTeam: false, canReadWikis: false })).toBe(
      false,
    );
    expect(shouldShowAiWikisNavigation({ isAiWikisEnabled: true, isPersonalTeam: true, canReadWikis: false })).toBe(
      true,
    );
  });

  it("hides AI Wikis when the feature is disabled", () => {
    expect(shouldShowAiWikisNavigation({ isAiWikisEnabled: false, isPersonalTeam: false, canReadWikis: true })).toBe(
      false,
    );
    expect(shouldShowAiWikisNavigation({ isAiWikisEnabled: false, isPersonalTeam: true, canReadWikis: true })).toBe(
      false,
    );
  });
});
