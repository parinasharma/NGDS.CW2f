// main.go - OLS regression of Indonesia CO2 on time (linear and quadratic).
// Setup (run once in your project folder):
//   go mod init heat-health
//   go get gonum.org/v1/gonum
// Run:  go run main.go   (NGDS_CO2_Indonesia.csv must be in the same folder)
package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Row holds one year of data plus the two features we create.
type Row struct {
	Year      int
	CO2       float64
	TimeIndex float64 // Year - 1970
	YearSq    float64 // TimeIndex squared (keeps numbers small and stable)
}

// loadRows reads the CSV, finds columns by header name and builds the features.
func loadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("file has no data rows")
	}

	// Map each header name to its column number (strip a possible BOM/spaces).
	col := map[string]int{}
	for i, h := range records[0] {
		col[strings.TrimSpace(strings.TrimPrefix(h, "\ufeff"))] = i
	}
	yearCol, ok := col["Year"]
	if !ok {
		return nil, fmt.Errorf("column Year not found")
	}
	co2Col, ok := col["CO2_Mt_CO2e"]
	if !ok {
		return nil, fmt.Errorf("column CO2_Mt_CO2e not found")
	}
	need := yearCol
	if co2Col > need {
		need = co2Col
	}

	var rows []Row
	for i, rec := range records[1:] {
		if len(rec) <= need {
			return nil, fmt.Errorf("row %d has too few columns", i+2)
		}
		year, err := strconv.Atoi(strings.TrimSpace(rec[yearCol]))
		if err != nil {
			return nil, fmt.Errorf("row %d: bad Year: %w", i+2, err)
		}
		co2, err := strconv.ParseFloat(strings.TrimSpace(rec[co2Col]), 64)
		if err != nil {
			return nil, fmt.Errorf("row %d: bad CO2: %w", i+2, err)
		}
		t := float64(year - 1970)
		rows = append(rows, Row{Year: year, CO2: co2, TimeIndex: t, YearSq: t * t})
	}
	return rows, nil
}

// splitRows puts years up to lastTrainYear in train and later years in test.
func splitRows(rows []Row, lastTrainYear int) (train, test []Row) {
	for _, r := range rows {
		if r.Year <= lastTrainYear {
			train = append(train, r)
		} else {
			test = append(test, r)
		}
	}
	return train, test
}

// pick returns one number from every row (used to build x and y slices).
func pick(rows []Row, get func(Row) float64) []float64 {
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i] = get(r)
	}
	return out
}

// fitOLS fits CO2 on Time_Index (linear) or Time_Index + Year_Sq (quadratic).
// It returns the coefficients: [intercept, b1] or [intercept, b1, b2].
func fitOLS(train []Row, quadratic bool) ([]float64, error) {
	y := pick(train, func(r Row) float64 { return r.CO2 })
	t := pick(train, func(r Row) float64 { return r.TimeIndex })

	if !quadratic {
		// Simple regression: gonum gives intercept (alpha) and slope (beta).
		alpha, beta := stat.LinearRegression(t, y, nil, false)
		return []float64{alpha, beta}, nil
	}

	// Quadratic: build the design matrix X with columns [1, Time_Index, Year_Sq].
	n := len(train)
	X := mat.NewDense(n, 3, nil)
	for i, r := range train {
		X.Set(i, 0, 1)
		X.Set(i, 1, r.TimeIndex)
		X.Set(i, 2, r.YearSq)
	}
	Y := mat.NewDense(n, 1, y)

	// Solve X * coef = Y in the least-squares sense (this is OLS).
	var coef mat.Dense
	if err := coef.Solve(X, Y); err != nil {
		return nil, fmt.Errorf("quadratic fit failed: %w", err)
	}
	return []float64{coef.At(0, 0), coef.At(1, 0), coef.At(2, 0)}, nil
}

// predict applies the fitted coefficients to one row.
func predict(coef []float64, r Row) float64 {
	p := coef[0] + coef[1]*r.TimeIndex
	if len(coef) == 3 {
		p += coef[2] * r.YearSq
	}
	return p
}

// metrics returns R-squared, RMSE and MAE for actual vs predicted values.
func metrics(actual, predicted []float64) (r2, rmse, mae float64) {
	mean := stat.Mean(actual, nil)
	var sse, sst, abs float64
	for i := range actual {
		e := actual[i] - predicted[i]
		sse += e * e
		sst += (actual[i] - mean) * (actual[i] - mean)
		abs += math.Abs(e)
	}
	n := float64(len(actual))
	return 1 - sse/sst, math.Sqrt(sse / n), abs / n
}

// main loads the data, fits both models and prints coefficients and the table.
func main() {
	rows, err := loadRows("NGDS_CO2_Indonesia.csv")
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	train, test := splitRows(rows, 2011)
	if len(train) == 0 || len(test) == 0 {
		fmt.Fprintln(os.Stderr, "error: train or test set is empty")
		os.Exit(1)
	}
	fmt.Printf("Train: %d rows (1970-2011)   Test: %d rows (2012-2022)\n\n", len(train), len(test))

	models := []struct {
		name      string
		quadratic bool
	}{
		{"Linear", false},
		{"Quadratic", true},
	}

	fmt.Printf("%-10s %-6s %9s %10s %10s\n", "Model", "Set", "R2", "RMSE", "MAE")
	for _, m := range models {
		coef, err := fitOLS(train, m.quadratic)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		fmt.Printf("\n%s coefficients: Intercept=%.4f  Time_Index=%.4f", m.name, coef[0], coef[1])
		if m.quadratic {
			fmt.Printf("  Year_Sq=%.4f", coef[2])
		}
		fmt.Println()

		sets := []struct {
			name string
			data []Row
		}{{"Train", train}, {"Test", test}}
		for _, s := range sets {
			actual := pick(s.data, func(r Row) float64 { return r.CO2 })
			pred := make([]float64, len(s.data))
			for i, r := range s.data {
				pred[i] = predict(coef, r)
			}
			r2, rmse, mae := metrics(actual, pred)
			fmt.Printf("%-10s %-6s %9.4f %10.3f %10.3f\n", m.name, s.name, r2, rmse, mae)
		}
	}
}
	


	