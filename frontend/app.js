const form = document.querySelector("#analysis-form");
const message = document.querySelector("#message");
const button = form.querySelector("button");
const statusElement = document.querySelector("#analysis-status");
const result = document.querySelector("#analysis-result");

button.disabled = false;

message.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();

        if (!button.disabled) {
            form.requestSubmit();
        }
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    result.replaceChildren();

    const text = message.value.trim();

    if (!text) {
        statusElement.textContent = "Digite uma mensagem para analisar.";
        message.focus();
        return;
    }

    button.disabled = true;
    statusElement.textContent = "Analisando...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error(`A análise falhou (HTTP ${response.status}).`);
        }

        const analysis = await response.json();
        const labels = {
            phishing: "Possível phishing detectado",
            legitimate: "Mensagem classificada como legítima",
            blocked: "Análise bloqueada pelo controle de qualidade do modelo",
        };

        const heading = document.createElement("h2");
        heading.textContent = labels[analysis.outcome] ?? "Resultado desconhecido";

        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Ver resposta completa";

        const output = document.createElement("pre");
        output.textContent = JSON.stringify(analysis, null, 2);

        details.append(summary, output);
        result.append(heading);

        for (const item of analysis.guidance?.results ?? []) {
            const sourceDetails = document.createElement("details");
            const sourceSummary = document.createElement("summary");
            sourceSummary.textContent = `Fonte: ${item.citation}`;

            const content = document.createElement("pre");
            content.textContent = item.content;

            sourceDetails.append(sourceSummary, content);
            result.append(sourceDetails);
        }

        result.append(details);

        statusElement.textContent = "Análise concluída.";
    } catch (error) {
        statusElement.textContent = error instanceof TypeError
          ? "Não foi possível conectar à API. Verifique se o serviço está iniciado e tente novamente."
          : error instanceof Error
            ? error.message
            : "Não foi possível concluir a análise. Tente Novamente.";
    } finally {
        button.disabled = false;
    }
});
