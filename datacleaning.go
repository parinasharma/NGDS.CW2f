// Data preprocessing for the NGDS Assignment 2 CO2 dataset (Indonesia).
// Pipeline: load -> validate -> clean missing values -> derive features -> write CSV.
package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	inputPath     = "data/raw/NGDS_CO2_Indonesia.csv"
	outputPath    = "data/processed/emissions_features.csv"
	lastTrainYear = 2011 // train 1970-2011, test 2012-2022 (time-ordered split)
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// Emission is one year of national CO2 emissions.
type Emission struct {
	Year    int
	CO2     float64 // Mt CO2e; NaN means the value is missing
	Imputed bool    // true if the value was filled in during cleaning
}

// CleanReport records what the cleaning step did, so the run is reproducible.
type CleanReport struct {
	RowsIn          int
	RowsOut         int
	DroppedLeading  int // missing values at the start (not extrapolated)
	DroppedTrailing int // missing values at the end (not extrapolated)
	Imputed         int // interior missing values filled by linear interpolation
}

// Feature is one cleaned year plus the derived regression columns.
type Feature struct {
	Year      int
	CO2       float64
	TimeIndex int     // Year minus the first year in the data
	YearSq    int     // TimeIndex squared
	Lag1      float64 // previous year's CO2; NaN for the first row
	Growth    float64 // CO2/Lag1 - 1; NaN for the first row
	LnCO2     float64 // natural log of CO2
	Decade    int     // e.g. 1990
	Set       string  // "Train" or "Test"
	Imputed   bool
}

// ---------------------------------------------------------------------------
// Step 1: load and validate
// ---------------------------------------------------------------------------

// LoadEmissions reads the CO2 CSV and validates its structure.
// Blank/NA values are kept as NaN so the cleaning step can decide what to do.
// Rows are returned sorted by year.
func LoadEmissions(path string) ([]Emission, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("%s has no data rows", path)
	}

	// Find columns by header name, so column order does not matter.
	col := map[string]int{}
	for i, h := range rows[0] {
		col[strings.TrimSpace(strings.TrimPrefix(h, "\ufeff"))] = i
	}
	yi, ok := col["Year"]
	if !ok {
		return nil, fmt.Errorf("column 'Year' not found")
	}
	ci, ok := col["CO2_Mt_CO2e"]
	if !ok {
		return nil, fmt.Errorf("column 'CO2_Mt_CO2e' not found")
	}

	var out []Emission
	seen := map[int]bool{}
	for n, r := range rows[1:] {
		line := n + 2 // line number in the file (header is line 1)
		if len(r) <= max(yi, ci) {
			return nil, fmt.Errorf("line %d: too few columns", line)
		}
		year, err := strconv.Atoi(strings.TrimSpace(r[yi]))
		if err != nil {
			return nil, fmt.Errorf("line %d: bad year %q: %w", line, r[yi], err)
		}
		if seen[year] {
			return nil, fmt.Errorf("line %d: duplicate year %d", line, year)
		}
		seen[year] = true

		raw := strings.TrimSpace(r[ci])
		if raw == "" || strings.EqualFold(raw, "NA") || strings.EqualFold(raw, "NaN") {
			out = append(out, Emission{Year: year, CO2: math.NaN()})
			continue
		}
		v, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return nil, fmt.Errorf("line %d: bad CO2 value %q: %w", line, raw, err)
		}
		if v < 0 {
			return nil, fmt.Errorf("line %d: negative CO2 value %v", line, v)
		}
		out = append(out, Emission{Year: year, CO2: v})
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Year < out[j].Year })
	return out, nil
}

// ---------------------------------------------------------------------------
// Step 2: clean missing values
// ---------------------------------------------------------------------------

// CleanEmissions validates years/values and handles missing CO2 values.
// Policy: interior gaps are linearly interpolated (and flagged Imputed);
// leading/trailing gaps are dropped, because extrapolating would invent data.
// Missing YEARS (gaps in the year sequence) are an error, since lag and
// growth features assume consecutive years.
func CleanEmissions(in []Emission) ([]Emission, CleanReport, error) {
	rep := CleanReport{RowsIn: len(in)}
	if len(in) == 0 {
		return nil, rep, fmt.Errorf("no rows to clean")
	}

	rows := make([]Emission, len(in))
	copy(rows, in)

	for i, r := range rows {
		if r.Year < 1900 || r.Year > 2100 {
			return nil, rep, fmt.Errorf("implausible year %d", r.Year)
		}
		if math.IsInf(r.CO2, 0) {
			return nil, rep, fmt.Errorf("year %d: infinite CO2 value", r.Year)
		}
		if i > 0 && r.Year != rows[i-1].Year+1 {
			return nil, rep, fmt.Errorf("year gap: %d follows %d", r.Year, rows[i-1].Year)
		}
	}

	// Drop leading and trailing missing values.
	start, end := 0, len(rows)
	for start < end && math.IsNaN(rows[start].CO2) {
		start++
		rep.DroppedLeading++
	}
	for end > start && math.IsNaN(rows[end-1].CO2) {
		end--
		rep.DroppedTrailing++
	}
	rows = rows[start:end]
	if len(rows) == 0 {
		return nil, rep, fmt.Errorf("all CO2 values are missing")
	}

	// Linearly interpolate interior gaps between the nearest real values.
	for i := range rows {
		if !math.IsNaN(rows[i].CO2) {
			continue
		}
		p := i - 1
		for p >= 0 && rows[p].Imputed {
			p--
		}
		n := i + 1
		for n < len(rows) && math.IsNaN(rows[n].CO2) {
			n++
		}
		if p < 0 || n >= len(rows) {
			return nil, rep, fmt.Errorf("cannot interpolate year %d", rows[i].Year)
		}
		frac := float64(rows[i].Year-rows[p].Year) / float64(rows[n].Year-rows[p].Year)
		rows[i].CO2 = rows[p].CO2 + (rows[n].CO2-rows[p].CO2)*frac
		rows[i].Imputed = true
		rep.Imputed++
	}

	rep.RowsOut = len(rows)
	return rows, rep, nil
}

// Print writes a short human-readable summary of the cleaning step.
func (r CleanReport) Print() {
	fmt.Printf("Cleaning report: in=%d out=%d dropped_leading=%d dropped_trailing=%d imputed=%d\n",
		r.RowsIn, r.RowsOut, r.DroppedLeading, r.DroppedTrailing, r.Imputed)
}

// ---------------------------------------------------------------------------
// Step 3: derive regression features
// ---------------------------------------------------------------------------

// BuildFeatures derives the regression columns from cleaned rows.
// Years up to and including lastTrainYear are labelled Train, later years Test
// (a time-ordered split, so the model never trains on the future).
func BuildFeatures(rows []Emission, lastTrainYear int) ([]Feature, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("no rows to build features from")
	}
	base := rows[0].Year
	out := make([]Feature, 0, len(rows))
	for i, r := range rows {
		if math.IsNaN(r.CO2) {
			return nil, fmt.Errorf("year %d: CO2 still missing; run CleanEmissions first", r.Year)
		}
		if r.CO2 <= 0 {
			return nil, fmt.Errorf("year %d: CO2 must be > 0 for ln(CO2), got %v", r.Year, r.CO2)
		}
		f := Feature{
			Year:      r.Year,
			CO2:       r.CO2,
			TimeIndex: r.Year - base,
			Lag1:      math.NaN(),
			Growth:    math.NaN(),
			LnCO2:     math.Log(r.CO2),
			Decade:    r.Year / 10 * 10,
			Set:       "Test",
			Imputed:   r.Imputed,
		}
		f.YearSq = f.TimeIndex * f.TimeIndex
		if r.Year <= lastTrainYear {
			f.Set = "Train"
		}
		if i > 0 {
			f.Lag1 = rows[i-1].CO2
			f.Growth = r.CO2/f.Lag1 - 1
		}
		out = append(out, f)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Step 4: write the processed CSV
// ---------------------------------------------------------------------------

func fmtFloat(v float64) string {
	if math.IsNaN(v) {
		return "" // blank cell = undefined value
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// WriteFeatures saves the processed dataset as CSV, creating folders if needed.
func WriteFeatures(path string, feats []Feature) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create folder for %s: %w", path, err)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	header := []string{"Year", "CO2_Mt_CO2e", "Time_Index", "Year_Sq", "CO2_Lag1",
		"CO2_Growth", "ln_CO2", "Decade", "Set", "Imputed"}
	if err := w.Write(header); err != nil {
		return err
	}
	for _, x := range feats {
		rec := []string{
			strconv.Itoa(x.Year), fmtFloat(x.CO2), strconv.Itoa(x.TimeIndex),
			strconv.Itoa(x.YearSq), fmtFloat(x.Lag1), fmtFloat(x.Growth),
			fmtFloat(x.LnCO2), strconv.Itoa(x.Decade), x.Set, strconv.FormatBool(x.Imputed),
		}
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Main: run the pipeline
// ---------------------------------------------------------------------------

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	raw, err := LoadEmissions(inputPath)
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d rows (%d-%d)\n", len(raw), raw[0].Year, raw[len(raw)-1].Year)

	clean, rep, err := CleanEmissions(raw)
	if err != nil {
		return err
	}
	rep.Print()

	feats, err := BuildFeatures(clean, lastTrainYear)
	if err != nil {
		return err
	}
	if err := WriteFeatures(outputPath, feats); err != nil {
		return err
	}
	fmt.Printf("Wrote %d rows to %s\n", len(feats), outputPath)
	return nil
}

