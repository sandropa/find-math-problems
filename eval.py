"""
Evaluate the AoPS problem finder agent against the benchmark problems in eval.tex.
Runs each problem through the agent and records whether a match was found.
Saves links, row indices, and actual AoPS problem text for manual verification.
"""
import os
import re
import sys
import json
import time
import signal
from contextlib import contextmanager

TIMEOUT_SECONDS = 90

@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from agent import build_agent
from tools import get_df, _clean_html, _make_link

PROBLEMS = {
    "A": [
        r"Pokazati da sistem jednadžbi $2ab = 6(a+b) - 13$ i $a^2 + b^2 = 4$ nema rješenja u skupu realnih brojeva.",
        r"Ako za realne brojeve $a, b, c$ vrijedi $a^2 + b^2 + c^2 = 2$, dokazati da je $ab + ac - bc \le 1$.",
        r"Neka su $a, b, c$ realni brojevi takvi da zbir nikoja dva nije jednak 1. Dokazati da je nemoguće da sva tri broja $\frac{ab}{a+b-1}, \frac{bc}{b+c-1}, \frac{ca}{c+a-1}$ pripadaju intervalu $(0,1)$.",
        r"Riješiti u skupu realnih brojeva jednadžbu $x\lfloor x \rfloor + \{x\} = 2018$.",
        r"Naći sve parove $(a, b)$ nenegativnih cijelih brojeva koji zadovoljavaju jednakost: $a + 2b - b^2 = \sqrt{2a + a^2 + |2a + 1 - 2b|}$.",
        r"Neka su $a, b, c$ pozitivni realni brojevi takvi da je $a^2 + b^2 + c^2 = 48$. Dokazati nejednakost $ac\sqrt{2b^3 + 16} + ba\sqrt{2c^3 + 16} + cb\sqrt{2a^3 + 16} \le 24^2$.",
        r"Neka su $a, b, c$ pozitivni realni brojevi takvi da je $ab + bc + ca = 3$. Dokazati da vrijedi nejednakost: $\frac{1}{1 + a^2(b+c)} + \frac{1}{1 + b^2(a+c)} + \frac{1}{1 + c^2(a+b)} \le \frac{a+b+c}{3abc}$.",
        r"Naći sve parove pozitivnih realnih brojeva $a, b$ takvih da vrijedi $(1+a)(8+b)(a+b) = 27ab$.",
        r"Neka je $n > 1$ prirodan broj a $a_1, a_2, \ldots, a_n$ cijeli brojevi, takvi da važi $a_1^2 + a_2^2 + \cdots + a_n^2 + n^3 \le (2n-1)(a_1 + a_2 + \cdots + a_n) + n^2$. Dokazati da su svi $a_i$ nenegativni i da $a_1 + a_2 + \cdots + a_n + n + 1$ nije potpun kvadrat.",
        r"Neka je $a_0 < a_1 < a_2 < \cdots$ beskonačni niz prirodnih brojeva. Dokazati da postoji jedinstven prirodan broj $n$ takav da je $a_n < \frac{a_0 + a_1 + a_2 + \cdots + a_n}{n} \le a_{n+1}$.",
    ],
    "N": [
        r"Dokazati da su svi brojevi oblika $10017, 100117, 1001117, \ldots$ djeljivi sa 53.",
        r"Neka je $n$ prirodan broj veći od 1. Dokazati da se broj $9^n$ može prikazati kao zbir kvadrata tri različita prirodna broja.",
        r"Odrediti najveći prirodan broj čije su sve cifre neparne, zbir cifara mu je 2015 i djeljiv je sa 7.",
        r"Naći sve trojke $(x, y, p)$, gdje su $x$ i $y$ cijeli brojevi a $p$ prost broj takvih da je $x^2 + 3xy + p^2y^2 = 12p$.",
        r"Neka su $x, y, z$ relativno prosti prirodni brojevi. Ako vrijedi $(y^2 - x^2) - (z^2 - y^2) = ((y - x) - (z - y))^2$ dokazati da su $x$ i $z$ potpuni kvadrati.",
        r"Naći sve prirodne brojeve $m$ i $n$ i sve proste brojeve $p \ge 5$ takve da je $m(4m^2 + m + 12) = 3(p^n - 1)$.",
        r"Primijetimo da brojevi $2, -3$ i $5$ imaju osobinu da je razlika bilo koja dva broja sadržilac trećeg broja. Neka različiti cijeli brojevi $a, b, c$ imaju istu osobinu. a) Dokazati da ne mogu svi biti pozitivni. b) Pretpostavimo da je $\gcd(a, b, c) = 1$. Dokazati da tada jedan od ova tri broja pripada skupu $\{1, 2, -1, -2\}$.",
        r"Odredi sve prirodne brojeve $(x, y, z)$ takve da je $(x+y)^2 - z^2 - 3x - y = 1$.",
        r"Dat je niz sa prvim članom koji je prirodan broj, takav da je svaki njegov član (osim prvog) jednak zbiru prethodnog člana sa brojem koji se dobije kada tom prethodnom članu obrnemo cifre. Dokazati da sedmi član tog niza nije prost.",
        r"Za prirodan broj $n$ kažemo da je šašav ako i samo ako postoje prirodni brojevi $a > 1$ i $b > 1$ takvi da je $n = a^b + b$. Da li postoji 2019 uzastopnih prirodnih brojeva među kojima je tačno 2017 šašavih brojeva?",
    ],
    "G": [
        r"U oštrokutnom trokutu $ABC$, točke $D, E, F$ su podnožja visina iz vrhova $A, B, C$, redom. Točke $P$ i $Q$ su projekcije točke $F$ na $AC$ i $BC$, redom. Dokazati da prava $PQ$ polovi duži $DF$ i $EF$.",
        r"Kružnice $k_1$ i $k_2$ s poluprecnicima $r_1$ i $r_2$ ($r_1 < r_2$) dodiruju se iznutra u točki $P$. Neka tangenta na $k_1$ koja je paralelna pravoj koja prolazi kroz centre ovih kružnica dodiruje $k_1$ u $R$, a siječe $k_2$ u $M$ i $N$. Dokazati da je $PR$ simetrala ugla $\angle MPN$.",
        r"Neka je točka $M$ proizvoljna točka simetrale ugla $\angle BAC$ trokuta $ABC$ koja se nalazi u trokutu. Prava $CM$ siječe krug opisan oko $ABC$ ponovo u $P$. Krug koji dodiruje pravu $CM$ u $M$ i prolazi kroz $A$ siječe $AB$ ponovo u $Q$ i krug opisan oko $ABC$ ponovo u $R$. Dokazati da su točke $P, Q, R$ kolinearne.",
        r"U oštrokutnom trokutu $ABC$ ($AC > BC$) točke $D$ i $E$ su podnožja visina iz $A$ i $B$, redom. Pri tome vrijedi $AB = 2DE$. Označimo centar opisane i centar upisane kružnice trokuta redom sa $O$ i $I$. Naći $\angle AIO$.",
        r"Neka je $ABCD$ konveksan četverougao, a $M$ i $N$ sredine stranica $AD$ i $BC$ redom. Pretpostavimo da je četverougao $ABNM$ tetivan i da je $AB$ tangenta na krug opisan oko trokuta $BMC$. Dokazati da je $AB$ tangenta i na krug opisan oko trokuta $AND$.",
        r"Dat je trokut $ABC$. Prava simetrična težišnoj duži iz $A$ u odnosu na simetralu ugla $\angle BAC$ siječe krug trokuta $ABC$ u $K$. Ako je $L$ središte duži $AK$, dokazati da je $\angle BLC = 2\angle BAC$.",
        r"Neka je $H$ ortocentar oštrokutnog trokuta $ABC$, a $M$ sredina stranice $BC$. Ako su $D$ i $E$ podnožja normala iz točke $H$ na simetralu unutrašnjeg i vanjskog ugla kod vrha $A$, dokazati da su točke $M$, $D$ i $E$ kolinearne.",
        r"Neka je $P$ točka na kružnici opisanoj oko trokuta $ABC$ na luku $\widehat{BC}$ na kojem nije točka $A$. Neka se prave $AB$ i $CP$ sijeku u točki $E$, a prave $AC$ i $BP$ sijeku u točki $F$. Ako simetrala stranice $\overline{AB}$ siječe $\overline{AC}$ u točki $K$, a simetrala stranice $\overline{AC}$ siječe $\overline{AB}$ u točki $J$, dokazati da je $\left(\frac{|CE|}{|BF|}\right)^2 = \frac{|AJ| \cdot |JE|}{|AK| \cdot |KF|}$.",
        r"Neka je $AK$ visina trokuta $ABC$, i neka upisana kružnica trokuta dodiruje stranice $BC, CA, AB$ redom u točkama $D, E, F$. Neka je $M$ točka na duži $AK$ takva da je $AM = AE$. Ako su $L$ i $N$ centri upisanih kružnica u trokutove $ABK$ i $ACK$, redom, dokazati da je četverougao $DLMN$ kvadrat.",
        r"Neka su $BB_1$ i $CC_1$ visine trokuta $\triangle ABC$, a $AD$ prečnik opisane kružnice tog trokuta. Prave $BB_1$ i $DC_1$ sijeku se u točki $E$, a prave $CC_1$ i $DB_1$ u $F$. Dokazati da je $\angle CAE = \angle BAF$.",
    ],
    "C": [
        r"Data je kvadratna jednadžba $ax^2 + bx + c = 0$. Igrači $A$ i $B$ igraju sljedeću igru: naizmjenično upisuju koeficijente ove kvadratne jednadžbe. Igrač $A$ upisuje dva koeficijenta, igrač $B$ jedan. Koeficijenti smiju biti svi realni brojevi osim 0. Igrač $A$ je pobijedio ukoliko su rješenja ove jednadžbe realna i suprotnog predznaka. Koji igrač ima pobjedničku strategiju?",
        r"Učenici su postrojeni u dva reda, tako da ispred svakog učenika stoji učenica koja je niža rastom od njega. Ako učenike postrojimo po veličini i ispred njih po veličini postrojimo učenice, dokazati da će opet ispred svakog učenika biti učenica koja je niža rastom od njega.",
        r"Neka je $S$ konačan neprazan skup prirodnih brojeva, takav da za sve $i$ i $j$ iz $S$ (ne nužno različite), vrijedi da $\frac{i+j}{\gcd(i,j)} \in S$. Naći sve takve skupove $S$.",
        r"Data su dva štapa, od kojih prvi ima dužinu $k$, a drugi dužinu $l$. Harun i Admir igraju sljedeću igru: najprije Harun razreže prvi štap na 3 dijela, a zatim Admir razreže drugi štap na 3 dijela. Ako se od novodobijenih 6 štapova mogu konstruisati dva trokuta, pobijedio je Admir. U zavisnosti od odnosa $k/l$ odrediti koji igrač ima pobjedničku strategiju.",
        r"Na horizontalnom štapu dužine 1 m je rasporedeno $n$ mrava, i svaki je okrenut lijevo ili desno. Svaki mrav se kreće brzinom 1 m/s. Ako mrav dođe do kraja štapa, on padne. Ako mrav sretne drugog mrava, njih dva se sudare i nastavljaju put u suprotnom smjeru. Dokazati da će za najviše jednu sekundu svi mravi sigurno pasti sa štapa.",
        r"Za koje prirodne brojeve $n$ možemo postaviti brojeve $1, 2, \ldots, n$ u vrhove pravilnog $n$-tougla, tako da za svaka tri vrha $A, B, C$ za koja vrijedi $AB = AC$, broj u vrhu $A$ bude ili manji od oba broja u vrhovima $B$ i $C$, ili veći od oba?",
        r"Dat je niz dužine 2019 koji se sastoji od prvih 2019 prirodnih brojeva poredanih u proizvoljnom redoslijedu. Uočimo prvi broj u tom nizu, neka je to $k$. Formiramo novi niz koji ima iste članove, samo što je prvih $k$ članova niza u obrnutom redoslijedu. Dokazati da će se pojaviti niz čiji je prvi element jedinica.",
        r"Pravougaona ploča dimenzija $m \times n$ podijeljena je na $m \cdot n$ jediničnih kvadratića. Na početku je u svakom kvadratiću upisan po jedan cijeli broj. U jednom koraku dozvoljeno je izabrati po jedan broj iz svakog reda i povećati sve izabrane brojeve za 1, ili izabrati iz svake kolone po jedan broj i smanjiti sve izabrane za 1. Da li je moguće dobiti ploču sa svim nulama ako je a) $m = 15, n = 20$; b) $m = 5, n = 11$? Za koje parove $(m, n)$ je moguće?",
        r"Naći maksimalan broj topova koji se mogu postaviti na šahovsku ploču formata $10 \times 10$, tako da među svakih 5 topova na ploči postoje dva koja se napadaju.",
        r"U ploču $2023 \times 2023$ su upisani prirodni brojevi $1, 2, 3, \ldots, 2023^2$ u nekom redoslijedu. Vuk može postaviti žeton u neko početno polje. U svakom potezu može pomjeriti žeton u bilo koje polje kvadrata $5 \times 5$ čiji je centar kvadrat gdje se žeton trenutno nalazi, pod uslovom da je u tom polju napisan veći broj. Odrediti najveći $k$ takav da Vuk može napraviti $k$ poteza neovisno od rasporeda.",
    ],
}

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "eval_checkpoint.json")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data)} problems already done.")
        return data
    return []


def save_checkpoint(results):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_eval(limit=None):
    results = load_checkpoint()
    done_labels = {r["label"] for r in results}

    agent = build_agent()

    all_problems = [(sec, i+1, p) for sec, ps in PROBLEMS.items() for i, p in enumerate(ps)]
    remaining = [(sec, i, p) for sec, i, p in all_problems if f"{sec}{i}" not in done_labels]
    if limit:
        remaining = remaining[:max(0, limit - len(done_labels))]

    total_all = len(all_problems)
    done = len(done_labels)

    for section, i, problem in remaining:
        label = f"{section}{i}"
        done += 1
        print(f"\n{'='*60}")
        print(f"[{done}/{total_all}] Running {label}...")
        print(f"{'='*60}")

        start = time.time()
        try:
            with time_limit(TIMEOUT_SECONDS):
                result = agent.invoke({"messages": [{"role": "user", "content": problem}]})
            response = result["messages"][-1].content
        except TimeoutError as e:
            result = {"messages": []}
            response = f"TIMEOUT: exceeded {TIMEOUT_SECONDS}s"
        except Exception as e:
            result = {"messages": []}
            response = f"ERROR: {e}"
        elapsed = time.time() - start

        # Determine if a match was found
        has_link = "artofproblemsolving.com" in response
        found = has_link and "no problems found" not in response.lower() and "could not find" not in response.lower() and "unable to find" not in response.lower() and "didn't find" not in response.lower()

        # Extract AoPS links from response for manual verification
        aops_links = list(set(re.findall(r'https?://artofproblemsolving\.com[^\s\)>\]]*', response)))

        # Extract row indices mentioned in tool calls (from agent messages)
        row_indices = []
        try:
            for msg in result["messages"]:
                content = msg.content if hasattr(msg, 'content') and isinstance(msg.content, str) else ""
                row_indices.extend(int(x) for x in re.findall(r'\[(\d+)\]', content))
        except Exception:
            pass
        row_indices = list(set(row_indices))

        # Look up actual problem text from dataset for each row index
        df = get_df()
        matched_problems = []
        for idx in row_indices[:5]:  # cap at 5
            if idx in df.index:
                row = df.loc[idx]
                matched_problems.append({
                    "row_index": idx,
                    "contest": str(row.get("contest", "")),
                    "name": str(row.get("name", "")),
                    "source_link": _make_link(row.get("source", "")),
                    "contest_link": _make_link(row.get("link", "")),
                    "problem_text": _clean_html(row["problem_html"])[:500],
                })

        results.append({
            "label": label,
            "section": section,
            "problem_num": i,
            "found": found,
            "has_link": has_link,
            "elapsed_s": round(elapsed, 1),
            "response": response,
            "aops_links": aops_links,
            "row_indices": row_indices,
            "matched_problems": matched_problems,
        })

        status = "FOUND" if found else "NOT FOUND"
        print(f"\n>>> {label}: {status} ({elapsed:.1f}s)")
        print(f"Response preview: {response[:200]}...")

        save_checkpoint(results)

    return results


def write_report(results):
    sections = {"A": "Algebra", "N": "Number Theory", "G": "Geometry", "C": "Combinatorics"}

    lines = []
    lines.append("# Evaluation Results")
    lines.append("")
    lines.append(f"Benchmark: `eval.tex` — 40 math competition problems from Bosnia and Herzegovina")
    lines.append(f"Agent: `anthropic/claude-haiku-4-5` via OpenRouter")
    lines.append("")

    # Summary
    total_found = sum(1 for r in results if r["found"])
    total = len(results)
    lines.append(f"## Summary: {total_found}/{total} found ({100*total_found/total:.0f}%)")
    lines.append("")

    lines.append("| Section | Found | Total | Rate |")
    lines.append("|---------|-------|-------|------|")
    for sec, name in sections.items():
        sec_results = [r for r in results if r["section"] == sec]
        sec_found = sum(1 for r in sec_results if r["found"])
        sec_total = len(sec_results)
        rate = f"{100*sec_found/sec_total:.0f}%" if sec_total else "N/A"
        lines.append(f"| {sec} ({name}) | {sec_found} | {sec_total} | {rate} |")
    lines.append("")

    avg_time = sum(r["elapsed_s"] for r in results) / len(results) if results else 0
    total_time = sum(r["elapsed_s"] for r in results)
    lines.append(f"Average time per problem: {avg_time:.1f}s")
    lines.append(f"Total time: {total_time:.0f}s")
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for sec, name in sections.items():
        lines.append(f"### {sec} — {name}")
        lines.append("")
        sec_results = [r for r in results if r["section"] == sec]
        for r in sec_results:
            status = "FOUND" if r["found"] else "NOT FOUND"
            lines.append(f"**{r['label']}** — {status} ({r['elapsed_s']}s)")
            lines.append("")
            # Agent response
            resp_lines = r["response"].strip().split("\n")
            for line in resp_lines:
                lines.append(f"> {line}")
            lines.append("")
            # Verification data
            if r["matched_problems"]:
                lines.append("<details><summary>Matched problems from dataset (for manual verification)</summary>")
                lines.append("")
                for mp in r["matched_problems"]:
                    lines.append(f"- **Row {mp['row_index']}**: {mp['contest']} — {mp['name']}")
                    lines.append(f"  - Problem link: {mp['source_link']}")
                    lines.append(f"  - Contest link: {mp['contest_link']}")
                    lines.append(f"  - Text: {mp['problem_text'][:300]}...")
                    lines.append("")
                lines.append("</details>")
                lines.append("")
            elif r["aops_links"]:
                lines.append(f"Links found: {', '.join(r['aops_links'])}")
                lines.append("")

    report = "\n".join(lines)

    with open("eval_results.md", "w") as f:
        f.write(report)
    print(f"\nReport written to eval_results.md")

    # Also save raw JSON
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Raw data written to eval_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N problems (for testing)")
    args = parser.parse_args()
    results = run_eval(limit=args.limit)
    write_report(results)
