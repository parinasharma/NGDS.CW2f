// Compare four regression models for Indonesia's annual CO2 emissions (1970-2022)
// on one shared chronological train/test split.
//
// Models:
//  1. Linear (OLS)          : CO2 ~ Time_Index
//  2. Quadratic (OLS)       : CO2 ~ Time_Index + Time_Index^2
//  3. Ridge Linear          : same predictor as (1), standardised, alpha chosen by time-series CV
//  4. Ridge Quadratic       : same predictors as (2), standardised, alpha chosen by time-series CV
//
// Only the Go standard library is used (no external modules needed).
//
// Usage (from the project root):
//
//	go run ./model_comparison -data NGDS_CO2_Indonesia.csv -out results
package main

import (
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
)

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

type dataset struct {
	years []int
	y     []float64
}

func loadData(path string) (*dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open data file: %w", err)
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse csv: %w", err)
	}
	if len(rows) < 2 {
		return nil, errors.New("csv has no data rows")
	}

	col := map[string]int{}
	for i, h := range rows[0] {
		col[strings.ToLower(strings.TrimSpace(strings.TrimPrefix(h, "\ufeff")))] = i
	}
	yi, ok1 := col["year"]
	ci, ok2 := col["co2_mt_co2e"]
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("csv must contain 'Year' and 'CO2_Mt_CO2e' columns, found %v", rows[0])
	}

	type rec struct {
		year int
		v    float64
	}
	var recs []rec
	for ln, r := range rows[1:] {
		if len(r) <= yi || len(r) <= ci {
			return nil, fmt.Errorf("row %d: too few columns", ln+2)
		}
		yr, err := strconv.Atoi(strings.TrimSpace(r[yi]))
		if err != nil {
			return nil, fmt.Errorf("row %d: bad year %q: %w", ln+2, r[yi], err)
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(r[ci]), 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("row %d: bad CO2 value %q", ln+2, r[ci])
		}
		recs = append(recs, rec{yr, v})
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].year < recs[j].year })

	d := &dataset{}
	for i, r := range recs {
		if i > 0 && r.year == recs[i-1].year {
			return nil, fmt.Errorf("duplicate year %d", r.year)
		}
		d.years = append(d.years, r.year)
		d.y = append(d.y, r.v)
	}
	return d, nil
}

// ---------------------------------------------------------------------------
// Linear algebra (small dense systems, Gaussian elimination with pivoting)
// ---------------------------------------------------------------------------

func solve(A [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	M := make([][]float64, n)
	for i := range M {
		M[i] = make([]float64, n+1)
		copy(M[i], A[i])
		M[i][n] = b[i]
	}
	for c := 0; c < n; c++ {
		p := c
		for r := c + 1; r < n; r++ {
			if math.Abs(M[r][c]) > math.Abs(M[p][c]) {
				p = r
			}
		}
		if math.Abs(M[p][c]) < 1e-12 {
			return nil, errors.New("singular or ill-conditioned matrix")
		}
		M[c], M[p] = M[p], M[c]
		for r := c + 1; r < n; r++ {
			f := M[r][c] / M[c][c]
			for k := c; k <= n; k++ {
				M[r][k] -= f * M[c][k]
			}
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := M[i][n]
		for k := i + 1; k < n; k++ {
			s -= M[i][k] * x[k]
		}
		x[i] = s / M[i][i]
	}
	return x, nil
}

// powers returns [t, t^2, ..., t^degree].
func powers(t float64, degree int) []float64 {
	out := make([]float64, degree)
	v := 1.0
	for k := 0; k < degree; k++ {
		v *= t
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

type predictFn func(t []float64) []float64

// fitter trains on (t, y) and returns a predictor plus coefficients on the
// ORIGINAL (unstandardised) Time_Index scale: [intercept, b1, b2, ...].
type fitter func(t, y []float64) (predictFn, []float64, error)

func polyPredict(coefs []float64) predictFn {
	return func(t []float64) []float64 {
		out := make([]float64, len(t))
		for i, ti := range t {
			s := coefs[0]
			for k, p := range powers(ti, len(coefs)-1) {
				s += coefs[k+1] * p
			}
			out[i] = s
		}
		return out
	}
}

// olsFitter: ordinary least squares via the normal equations (no scaling, so
// coefficients are directly comparable with a conventional OLS fit).
func olsFitter(degree int) fitter {
	return func(t, y []float64) (predictFn, []float64, error) {
		n, p := len(t), degree+1
		if n <= p {
			return nil, nil, fmt.Errorf("need more than %d samples, got %d", p, n)
		}
		XtX := make([][]float64, p)
		for i := range XtX {
			XtX[i] = make([]float64, p)
		}
		Xty := make([]float64, p)
		for i := 0; i < n; i++ {
			row := append([]float64{1}, powers(t[i], degree)...)
			for a := 0; a < p; a++ {
				Xty[a] += row[a] * y[i]
				for b := 0; b < p; b++ {
					XtX[a][b] += row[a] * row[b]
				}
			}
		}
		beta, err := solve(XtX, Xty)
		if err != nil {
			return nil, nil, err
		}
		return polyPredict(beta), beta, nil
	}
}

// ridgeFitter: features are standardised with the TRAINING mean/std only, the
// target is centred with the training mean, and the intercept is NOT penalised.
//
//	w = (Z'Z + alpha*I)^-1 Z'(y - ybar)
func ridgeFitter(degree int, alpha float64) fitter {
	return func(t, y []float64) (predictFn, []float64, error) {
		n := len(t)
		if n <= degree+1 {
			return nil, nil, fmt.Errorf("need more than %d samples, got %d", degree+1, n)
		}
		p := degree
		X := make([][]float64, n)
		for i := range X {
			X[i] = powers(t[i], degree)
		}
		mean := make([]float64, p)
		std := make([]float64, p)
		for j := 0; j < p; j++ {
			for i := 0; i < n; i++ {
				mean[j] += X[i][j]
			}
			mean[j] /= float64(n)
			for i := 0; i < n; i++ {
				d := X[i][j] - mean[j]
				std[j] += d * d
			}
			std[j] = math.Sqrt(std[j] / float64(n))
			if std[j] < 1e-12 {
				return nil, nil, fmt.Errorf("feature %d has zero variance in training data", j+1)
			}
		}
		yMean := 0.0
		for _, v := range y {
			yMean += v
		}
		yMean /= float64(n)

		A := make([][]float64, p)
		for a := range A {
			A[a] = make([]float64, p)
		}
		rhs := make([]float64, p)
		for i := 0; i < n; i++ {
			z := make([]float64, p)
			for j := 0; j < p; j++ {
				z[j] = (X[i][j] - mean[j]) / std[j]
			}
			for a := 0; a < p; a++ {
				rhs[a] += z[a] * (y[i] - yMean)
				for b := 0; b < p; b++ {
					A[a][b] += z[a] * z[b]
				}
			}
		}
		for a := 0; a < p; a++ {
			A[a][a] += alpha
		}
		w, err := solve(A, rhs)
		if err != nil {
			return nil, nil, err
		}

		pred := func(tt []float64) []float64 {
			out := make([]float64, len(tt))
			for i, ti := range tt {
				x := powers(ti, degree)
				s := yMean
				for j := 0; j < p; j++ {
					s += w[j] * (x[j] - mean[j]) / std[j]
				}
				out[i] = s
			}
			return out
		}
		// coefficients mapped back to the original scale (reporting only)
		coefs := make([]float64, p+1)
		coefs[0] = yMean
		for j := 0; j < p; j++ {
			coefs[j+1] = w[j] / std[j]
			coefs[0] -= coefs[j+1] * mean[j]
		}
		return pred, coefs, nil
	}
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

type metrics struct{ R2, RMSE, MAE float64 }

func evaluate(y, p []float64) metrics {
	n := float64(len(y))
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= n
	var ssRes, ssTot, abs float64
	for i := range y {
		e := y[i] - p[i]
		ssRes += e * e
		abs += math.Abs(e)
		ssTot += (y[i] - mean) * (y[i] - mean)
	}
	r2 := math.NaN()
	if ssTot > 0 {
		r2 = 1 - ssRes/ssTot
	}
	return metrics{R2: r2, RMSE: math.Sqrt(ssRes / n), MAE: abs / n}
}

// ---------------------------------------------------------------------------
// Time-series cross-validation (expanding window, training data only)
// ---------------------------------------------------------------------------

// cvRMSE: fold k trains on t[:s] and validates on t[s:s+valSize], with s
// starting at minTrain and advancing by valSize. Scaling happens inside the
// fitter, so each fold's scaler only ever sees that fold's training rows.
func cvRMSE(f fitter, t, y []float64, minTrain, valSize int) (float64, int, error) {
	sum, folds := 0.0, 0
	for s := minTrain; s+valSize <= len(t); s += valSize {
		pred, _, err := f(t[:s], y[:s])
		if err != nil {
			return 0, 0, err
		}
		sum += evaluate(y[s:s+valSize], pred(t[s:s+valSize])).RMSE
		folds++
	}
	if folds == 0 {
		return 0, 0, errors.New("no CV folds: training set too small for -min-train/-val-size")
	}
	return sum / float64(folds), folds, nil
}

type alphaScore struct{ Alpha, CVRMSE float64 }

func selectAlpha(degree int, grid []float64, t, y []float64, minTrain, valSize int) (float64, []alphaScore, error) {
	best, bestScore := math.NaN(), math.Inf(1)
	var table []alphaScore
	for _, a := range grid {
		score, _, err := cvRMSE(ridgeFitter(degree, a), t, y, minTrain, valSize)
		if err != nil {
			return 0, nil, err
		}
		table = append(table, alphaScore{a, score})
		if score < bestScore {
			best, bestScore = a, score
		}
	}
	return best, table, nil
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

type result struct {
	Name, Predictors string
	Alpha            float64 // NaN for OLS
	CV               float64
	Coefs            []float64
	Train, Test      metrics
	TestPred         []float64
	Fit              fitter // refit on any (t, y); used by forcast.go
}

func fmtF(v float64) string {
	if math.IsNaN(v) {
		return ""
	}
	return strconv.FormatFloat(v, 'f', 6, 64)
}

func run() error {
	dataPath := flag.String("data", "NGDS_CO2_Indonesia.csv", "path to the CSV (needs Year and CO2_Mt_CO2e columns)")
	outDir := flag.String("out", "results", "output directory for CSV files")
	trainFrac := flag.Float64("train-frac", 0.8, "fraction of years (earliest) used for training")
	minTrain := flag.Int("min-train", 20, "smallest training window used in time-series CV")
	valSize := flag.Int("val-size", 5, "validation window length used in time-series CV")
	flag.Parse()

	d, err := loadData(*dataPath)
	if err != nil {
		return err
	}
	n := len(d.y)
	nTrain := int(math.Floor(*trainFrac * float64(n)))
	if nTrain < *minTrain+*valSize || n-nTrain < 2 {
		return fmt.Errorf("bad split: %d rows -> %d train / %d test", n, nTrain, n-nTrain)
	}

	// Time_Index = Year - first year (0,1,2,...). Same chronological split for ALL models.
	t := make([]float64, n)
	for i, yr := range d.years {
		t[i] = float64(yr - d.years[0])
	}
	tTr, yTr := t[:nTrain], d.y[:nTrain]
	tTe, yTe := t[nTrain:], d.y[nTrain:]

	fmt.Printf("Data: %d rows, %d-%d\n", n, d.years[0], d.years[n-1])
	fmt.Printf("Train: %d rows (%d-%d)   Test: %d rows (%d-%d)\n\n",
		nTrain, d.years[0], d.years[nTrain-1], n-nTrain, d.years[nTrain], d.years[n-1])

	var grid []float64
	for e := -4.0; e <= 3.0001; e += 0.5 {
		grid = append(grid, math.Pow(10, e))
	}

	type spec struct {
		name, predictors string
		degree           int
		ridge            bool
	}
	specs := []spec{
		{"Linear (OLS)", "Time_Index", 1, false},
		{"Quadratic (OLS)", "Time_Index, Time_Index^2", 2, false},
		{"Ridge Linear", "Time_Index (standardised)", 1, true},
		{"Ridge Quadratic", "Time_Index, Time_Index^2 (standardised)", 2, true},
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	var results []result
	var alphaRows [][]string
	alphaRows = append(alphaRows, []string{"Model", "Alpha", "CV_RMSE_train_only"})

	for _, s := range specs {
		r := result{Name: s.name, Predictors: s.predictors, Alpha: math.NaN()}
		var f fitter
		if s.ridge {
			best, table, err := selectAlpha(s.degree, grid, tTr, yTr, *minTrain, *valSize)
			if err != nil {
				return fmt.Errorf("%s alpha selection: %w", s.name, err)
			}
			r.Alpha = best
			for _, a := range table {
				alphaRows = append(alphaRows, []string{s.name, fmtF(a.Alpha), fmtF(a.CVRMSE)})
			}
			f = ridgeFitter(s.degree, best)
		} else {
			f = olsFitter(s.degree)
		}
		r.Fit = f

		cv, _, err := cvRMSE(f, tTr, yTr, *minTrain, *valSize)
		if err != nil {
			return fmt.Errorf("%s CV: %w", s.name, err)
		}
		r.CV = cv

		pred, coefs, err := f(tTr, yTr) // final fit: training data only
		if err != nil {
			return fmt.Errorf("%s fit: %w", s.name, err)
		}
		r.Coefs = coefs
		r.Train = evaluate(yTr, pred(tTr))
		r.TestPred = pred(tTe) // test set touched only here, for evaluation
		r.Test = evaluate(yTe, r.TestPred)
		results = append(results, r)
	}

	bestTest, bestCV := 0, 0
	for i, r := range results {
		if r.Test.RMSE < results[bestTest].Test.RMSE {
			bestTest = i
		}
		if r.CV < results[bestCV].CV {
			bestCV = i
		}
	}

	// ---- console report ----
	for _, r := range results {
		fmt.Printf("%-16s coefficients (original Time_Index scale): ", r.Name)
		for k, c := range r.Coefs {
			if k > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("b%d=%.4f", k, c)
		}
		if !math.IsNaN(r.Alpha) {
			fmt.Printf("   [alpha=%g]", r.Alpha)
		}
		fmt.Println()
	}
	fmt.Println()

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Model\tAlpha\tCV RMSE (train only)\tTrain R2\tTest R2\tTest RMSE\tTest MAE\tR2 gap")
	for _, r := range results {
		al := "-"
		if !math.IsNaN(r.Alpha) {
			al = fmt.Sprintf("%g", r.Alpha)
		}
		fmt.Fprintf(tw, "%s\t%s\t%.3f\t%.4f\t%.4f\t%.3f\t%.3f\t%.4f\n",
			r.Name, al, r.CV, r.Train.R2, r.Test.R2, r.Test.RMSE, r.Test.MAE, r.Train.R2-r.Test.R2)
	}
	tw.Flush()

	fmt.Printf("\nBest on TEST set (lowest test RMSE): %s  (RMSE=%.3f, MAE=%.3f, R2=%.4f)\n",
		results[bestTest].Name, results[bestTest].Test.RMSE, results[bestTest].Test.MAE, results[bestTest].Test.R2)
	fmt.Printf("Best on training-only time-series CV : %s  (CV RMSE=%.3f)\n", results[bestCV].Name, results[bestCV].CV)
	if bestTest != bestCV {
		fmt.Println("Note: the two criteria disagree; see the explanation below.")
	}

	widest := 0
	for i, r := range results {
		if r.Train.R2-r.Test.R2 > results[widest].Train.R2-results[widest].Test.R2 {
			widest = i
		}
	}
	fmt.Printf("\nLargest train-to-test R2 drop: %s (%.4f -> %.4f)\n",
		results[widest].Name, results[widest].Train.R2, results[widest].Test.R2)
	fmt.Println(`
Overfitting vs generalisation
  * Training R2 only measures how well a model reproduces data it was fitted on;
    it never decreases when a model gets more flexible, so it cannot rank models.
  * Overfitting shows up as a large drop from training to test performance.
  * Here the test years (the latest 20%) lie OUTSIDE the training range and at much
    higher emission levels, so the test is an extrapolation. A model that fits the
    past well can still fail on it, and R2 on a short, high-level test window
    is harsh (it can be negative even when errors are moderate), so RMSE/MAE are
    reported alongside it.
  * Ridge shrinks coefficients to reduce variance. With one predictor (or two very
    correlated ones) and 42 training rows there is little variance to remove, so
    expect ridge to behave much like its OLS counterpart; check the alpha selected.
  * Models are compared on test RMSE/MAE/R2 above; the training-only CV RMSE is
    shown as an independent check that does not use the test set.`)

	// ---- CSV output ----
	cmpPath := filepath.Join(*outDir, "model_comparison.csv")
	if err := writeCSV(cmpPath, func() [][]string {
		rows := [][]string{{"Model", "Predictors", "Alpha", "CV_RMSE_train_only", "Train_R2", "Train_RMSE", "Train_MAE",
			"Test_R2", "Test_RMSE", "Test_MAE", "R2_Gap_Train_minus_Test", "Best_On_Test"}}
		for i, r := range results {
			rows = append(rows, []string{r.Name, r.Predictors, fmtF(r.Alpha), fmtF(r.CV),
				fmtF(r.Train.R2), fmtF(r.Train.RMSE), fmtF(r.Train.MAE),
				fmtF(r.Test.R2), fmtF(r.Test.RMSE), fmtF(r.Test.MAE),
				fmtF(r.Train.R2 - r.Test.R2), strconv.FormatBool(i == bestTest)})
		}
		return rows
	}()); err != nil {
		return err
	}
	if err := writeCSV(filepath.Join(*outDir, "alpha_cv_results.csv"), alphaRows); err != nil {
		return err
	}
	predRows := [][]string{{"Year", "Actual"}}
	for _, r := range results {
		predRows[0] = append(predRows[0], r.Name)
	}
	for i := range yTe {
		row := []string{strconv.Itoa(d.years[nTrain+i]), fmtF(yTe[i])}
		for _, r := range results {
			row = append(row, fmtF(r.TestPred[i]))
		}
		predRows = append(predRows, row)
	}
	if err := writeCSV(filepath.Join(*outDir, "test_predictions.csv"), predRows); err != nil {
		return err
	}
	fmt.Printf("\nSaved: %s, alpha_cv_results.csv, test_predictions.csv (in %s)\n", cmpPath, *outDir)
	return nil
}

func writeCSV(path string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	w := csv.NewWriter(f)
	if err := w.WriteAll(rows); err != nil {
		f.Close()
		return fmt.Errorf("write %s: %w", path, err)
	}
	return f.Close()
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
