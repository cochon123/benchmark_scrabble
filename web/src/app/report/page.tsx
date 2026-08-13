import type { Metadata } from "next";

import styles from "./report.module.css";

export const metadata: Metadata = {
  title: "Qwen3-4B Scrabble v3 report",
  description: "Training and held-out diagnostic results for the general Scrabble v3 experiment.",
};

const evaluations = [
  { label: "Base", score: 0, legal: 0, complete: 0 },
  { label: "Step 100", score: 0, legal: 0, complete: 100 },
  { label: "Step 500", score: 0, legal: 0, complete: 100 },
  { label: "Step 1,000", score: 0, legal: 0, complete: 100 },
  { label: "Step 1,500", score: 0, legal: 0, complete: 100 },
  { label: "Step 2,000", score: 0, legal: 0, complete: 100 },
  { label: "Step 2,500", score: 0, legal: 0, complete: 100 },
  { label: "Final · 2,735", score: 0, legal: 0, complete: 100 },
];

const loss = [
  { name: "Step 10", value: "13.9846", note: "initial weighted loss" },
  { name: "Step 100", value: "1.8608", note: "format learned" },
  { name: "Step 500", value: "0.9831", note: "still 0/5 legal" },
  { name: "Step 1,000", value: "0.7897", note: "still 0/5 legal" },
  { name: "Step 1,500", value: "0.6631", note: "still 0/5 legal" },
  { name: "Step 2,000", value: "0.6192", note: "still 0/5 legal" },
  { name: "Final", value: "0.6039", note: "validation loss 0.07837" },
];

export default function ReportPage() {
  return (
    <article className={styles.report}>
      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <p className={styles.kicker}>General Scrabble v3 · August 4, 2026</p>
          <h2>It learned the format. It did not yet learn reliable play.</h2>
          <p className={styles.lede}>
            Qwen3-4B completed <strong>2,735 optimizer steps</strong> on 21,874 clean training records. On a matched 20-board held-out comparison, v3 slightly exceeded v2 in points, but both produced only <strong>one legal move</strong>.
          </p>
          <div className={styles.actions}>
            <a className={styles.primaryAction} href="https://huggingface.co/Cochon123/Qwen3-4B-Scrabble-General-v3-work" target="_blank" rel="noreferrer">Model and raw traces ↗</a>
            <a className={styles.secondaryAction} href="#trace">See recovery trace</a>
          </div>
        </div>
        <div className={styles.scoreCard} aria-label="V3 held-out test score: 2.22 percent">
          <span className={styles.scoreLabel}>V3 · 20-board test</span>
          <strong className={styles.score}>2.22<span>%</span></strong>
          <div className={styles.scoreRule} />
          <span>14 / 630 points · 1 / 20 legal · 0 optimal</span>
        </div>
      </section>

      <section className={styles.metrics} aria-label="Experiment highlights">
        <div><span>Base model</span><strong>Qwen3-4B</strong><small>Thinking 2507</small></div>
        <div><span>Training</span><strong>2,735 steps</strong><small>L40S · $5.184</small></div>
        <div><span>Validation</span><strong>0.07837</strong><small>1,710 records</small></div>
        <div><span>Matched test</span><strong>20 games</strong><small>T4 · $0.300</small></div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeading}>
          <div><p className={styles.kicker}>Matched comparison</p><h3>V3 is not worse—but both are weak.</h3></div>
          <p>Twenty untouched positions from twenty games, balanced across plies 0/4/7/10/14. Greedy, seed 3407, 512 tokens, two attempts.</p>
        </div>
        <div className={styles.tableWrap}><table><thead><tr><th>Model</th><th>Points</th><th>Score</th><th>Legal</th><th>Recovery</th></tr></thead><tbody><tr><td>V2 sparse</td><td>10/630</td><td>1.59%</td><td>1/20</td><td>0</td></tr><tr><td>V3 sparse</td><td>14/630</td><td>2.22%</td><td>1/20</td><td>1</td></tr><tr><td>V3 dense</td><td>14/630</td><td>2.22%</td><td>1/20</td><td>0</td></tr></tbody></table></div>
        <p className={styles.bodyCopy}>Every legal move was on an empty opening board. Every non-empty-board case was illegal, so occupied-tile grounding and cross-word verification remain the dominant failure.</p>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeading}>
          <div><p className={styles.kicker}>Behavioral evolution</p><h3>A flat legal-move curve</h3></div>
          <p>Same five validation-only boards at every checkpoint, greedy decoding, seed 3407, two validator-feedback attempts, and a corrected 512-token ceiling.</p>
        </div>
        <div className={styles.evaluationList}>
          {evaluations.map((item) => (
            <div className={styles.evaluationRow} key={item.label}>
              <div className={styles.evaluationMeta}>
                <strong>{item.label}</strong>
                <span>{item.legal}% legal · {item.complete}% complete structured responses</span>
              </div>
              <div className={styles.barTrack}><div className={styles.bar} style={{ width: `${Math.max(item.score, 0.5)}%` }} /></div>
              <strong className={styles.percent}>{item.score}%</strong>
            </div>
          ))}
        </div>
        <div className={styles.target}><span>What changed?</span><strong>0% → 100%</strong><span>Answer completion improved; strict Scrabble legality did not.</span></div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeading}>
          <div><p className={styles.kicker}>Optimization</p><h3>Loss fell 95.7%</h3></div>
          <p>Weighted training loss strongly improved while the behavioral diagnostic did not. This is the central result of v3.</p>
        </div>
        <div className={styles.stageGrid}>
          {loss.map((point, index) => (
            <div className={styles.stageCard} key={point.name}>
              <span className={styles.stageNumber}>{String(index + 1).padStart(2, "0")}</span>
              <h4>{point.name}</h4>
              <p>{point.note}</p>
              <dl><div><dt>Weighted loss</dt><dd>{point.value}</dd></div></dl>
            </div>
          ))}
        </div>
        <div className={styles.actions}>
          <a className={styles.secondaryAction} href="https://huggingface.co/Cochon123/Qwen3-4B-Scrabble-General-v3-work/blob/main/evaluations/evolution/training_loss_chart.svg" target="_blank" rel="noreferrer">Open full loss chart ↗</a>
          <a className={styles.secondaryAction} href="https://huggingface.co/Cochon123/Qwen3-4B-Scrabble-General-v3-work/blob/main/evaluations/evolution/evolution_chart.svg" target="_blank" rel="noreferrer">Open evolution chart ↗</a>
        </div>
      </section>

      <section className={styles.twoColumn} id="trace">
        <div className={styles.section}>
          <p className={styles.kicker}>V3 reasoning trace · successful recovery</p>
          <h3>Feedback fixed one opening move.</h3>
          <div className={styles.trace}>
            <div><strong>Attempt 1</strong><span className={styles.traceBad}>Rejected · misses center</span></div>
            <pre>BOP down{`\n`}B/O/P at (7,6), (8,6), (9,6)</pre>
            <div><strong>Attempt 2</strong><span className={styles.traceGood}>Legal · exact score 14</span></div>
            <pre>BOP across{`\n`}B/O/P at (7,5), (7,6), (7,7)</pre>
          </div>
          <p className={styles.bodyCopy}>The retry correctly changed orientation and covered the center. The optimum was 24, so this proves real rule recovery—not strong move search.</p>
        </div>
        <aside className={styles.caveat}>
          <span className={styles.caveatIcon}>!</span>
          <div><p className={styles.kicker}>V3 diagnosis</p><h3>Fluent imitation is not execution.</h3><p>V3 final responses described reconstruction and candidate checks, then emitted valid JSON. The moves still failed on invalid words, cross-words, rack letters, collisions, connectivity, center coverage, or bounds. Repeated bad candidates across checkpoints suggest template imitation and weak board grounding. No v3 retry recovered a board.</p></div>
        </aside>
      </section>

      <section className={styles.section} id="method">
        <div className={styles.sectionHeading}>
          <div><p className={styles.kicker}>Integrity and incident history</p><h3>The negative result is reproducible.</h3></div>
          <p>Five boards came from distinct held-out validation games and were never used for training or checkpoint selection.</p>
        </div>
        <p className={styles.bodyCopy}>The first evaluation omitted its lexicon. A second completed seven labels but exposed 256-token mid-JSON truncation and a missing final-tokenizer chat template. Both failures, costs, and partial outputs were preserved. The corrected run passed 33 tests, completed all eight labels, and every downloaded result matched its completion SHA-256 manifest.</p>
        <div className={styles.actions}>
          <a className={styles.primaryAction} href="https://huggingface.co/jobs/Cochon123/6a71b8406b79c09949c21a78" target="_blank" rel="noreferrer">Completed evaluation job ↗</a>
          <a className={styles.secondaryAction} href="https://huggingface.co/Cochon123/Qwen3-4B-Scrabble-General-v3-work/tree/main/evaluations/evolution" target="_blank" rel="noreferrer">All JSON, CSV, SVG and traces ↗</a>
        </div>
      </section>

      <footer className={styles.footer}>
        <p>Qwen3-4B Scrabble v3 · dense boards · greedy · 512 tokens · two attempts · seed 3407</p>
        <span>Five boards are diagnostic evidence, not a full benchmark estimate.</span>
      </footer>
    </article>
  );
}
