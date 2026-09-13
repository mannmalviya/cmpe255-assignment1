import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.describe("Tempo task workspace", () => {
  test("supports the complete task lifecycle without console errors", async ({ page, request }, testInfo) => {
    const browserErrors: string[] = [];
    page.on("console", (message) => { if (message.type() === "error") browserErrors.push(message.text()); });
    page.on("pageerror", (error) => browserErrors.push(error.message));

    const health = await request.get("/api/health");
    expect(health.ok()).toBeTruthy();
    expect(await health.json()).toMatchObject({ status: "ok", database: "sqlite", integrity: "ok" });

    await page.goto("/");
    await expect(page.getByRole("heading", { name: "My tasks", level: 1 })).toBeVisible();
    const title = `Browser verified ${Date.now()}`;
    await page.getByLabel("Quick-add a task").fill(title);
    await page.getByRole("button", { name: "Add task" }).click();
    await expect(page.getByText(title, { exact: true })).toBeVisible();

    await page.getByText(title, { exact: true }).click();
    await expect(page.getByRole("dialog", { name: "Edit task" })).toBeVisible();
    await page.getByLabel(/Notes/).fill("Validated from a real Chrome session");
    await page.getByRole("dialog").locator("select").selectOption("high");
    await page.getByLabel(/Tags/).fill("browser, verified");
    await page.getByRole("button", { name: "Save changes" }).click();
    await expect(page.getByText("Validated from a real Chrome session")).toBeVisible();

    await page.getByLabel("Search tasks").fill("browser verified");
    await expect(page.getByTestId("task-row")).toHaveCount(1);
    await page.getByLabel(`Complete ${title}`).click();
    await expect(page.getByText(title, { exact: true })).toBeHidden();
    await page.getByLabel("Clear search").click();
    if (testInfo.project.name.includes("mobile")) {
      await page.getByRole("button", { name: /Views/ }).click();
    }
    await page.getByRole("button", { name: /Completed/ }).click();
    await expect(page.getByText(title, { exact: true })).toBeVisible();

    await page.getByRole("button", { name: `Actions for ${title}` }).click();
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(page.getByText(title, { exact: true })).toBeHidden();
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.getByText(title, { exact: true })).toBeVisible();

    const accessibility = await new AxeBuilder({ page }).analyze();
    expect(accessibility.violations).toEqual([]);
    expect(browserErrors).toEqual([]);
    await page.screenshot({ path: `docs/screenshots/${testInfo.project.name}-workspace.png`, fullPage: true });

    const persisted = await (await request.get("/api/tasks")).json() as { tasks: { id: number; title: string }[] };
    for (const task of persisted.tasks.filter((item) => item.title === title)) {
      expect((await request.delete(`/api/tasks/${task.id}`)).ok()).toBeTruthy();
    }
  });

  test("adapts navigation and editor to the active viewport", async ({ page }, testInfo) => {
    const browserErrors: string[] = [];
    page.on("console", (message) => { if (message.type() === "error") browserErrors.push(message.text()); });
    page.on("pageerror", (error) => browserErrors.push(error.message));
    await page.goto("/");

    await page.getByRole("button", { name: "Toggle color theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

    if (testInfo.project.name.includes("mobile")) {
      await expect(page.getByRole("button", { name: /Views/ })).toBeVisible();
      await page.getByRole("button", { name: /Views/ }).click();
      await expect(page.getByLabel("Task views")).toBeInViewport();
      await page.getByRole("button", { name: /Today/ }).click();
      await expect(page.getByRole("heading", { name: "Today", level: 1 })).toBeVisible();
    } else {
      await expect(page.getByLabel("Task views")).toBeVisible();
    }

    if (testInfo.project.name.includes("mobile")) {
      await page.getByRole("button", { name: /Views/ }).click();
    }
    await page.getByRole("button", { name: /New task/ }).click();
    await expect(page.getByRole("dialog", { name: "Create a task" })).toBeVisible();
    const accessibility = await new AxeBuilder({ page }).analyze();
    expect(accessibility.violations).toEqual([]);
    await page.getByRole("button", { name: "Close task editor" }).click();
    expect(browserErrors).toEqual([]);
    await page.screenshot({ path: `docs/screenshots/${testInfo.project.name}-responsive.png`, fullPage: true });
  });
});
