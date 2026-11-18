// visualise/main.go - publication-style charts for the Indonesia CO2 regression project.
// Uses your CSV, refits the same Linear and Quadratic OLS models (train 1970-2011), and
// saves PNG figures into ./figures. Nothing is typed in by hand: every plotted number
// comes from the CSV or from the fitted models.
//
// Setup (once):  go get gonum.org/v1/gonum gonum.org/v1/plot
// Run (from the project folder that holds the CSV):
//
//	go run ./visualise                      (uses NGDS_CO2_Indonesia.csv)
//	go run ./visualise your_file.csv        (or give another file name)
package main

import (
	"encoding/csv"
	"fmt"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	baseYear      = 1970 // Time_Index = Year - baseYear
	lastTrainYear = 2011 // Train: 1970-2011, Test: 2012-2022
	imgWidth      = 10 * vg.Inch
	imgHeight     = 6 * vg.Inch
	imgDPI        = 200 // 10x6 inches at 200 dpi = 2000x1200 pixels (sharp on slides)
	yLabelCO2     = "CO₂ emissions (Mt CO₂e)"
)

// Colour-blind-friendly palette.
var (
	colActual = color.NRGBA{R: 34, G: 34, B: 34, A: 255}
	colLinear = color.NRGBA{R: 213, G: 94, B: 0, A: 255}
	colQuad   = color.NRGBA{R: 0, G: 114, B: 178, A: 255}
	colGrid   = color.NRGBA{R: 225, G: 225, B: 225, A: 255}
	colZero   = color.NRGBA{R: 90, G: 90, B: 90, A: 255}
	colShade  = color.NRGBA{R: 120, G: 120, B: 120, A: 45}
	colFaint  = color.NRGBA{R: 150, G: 150, B: 150, A: 255}
)

// Row holds one year of data plus the two features used by the models.
type Row struct {
	Year      int
	CO2       float64
	TimeIndex float64 // Year - 1970
	YearSq    float64 // Time_Index squared
}

// model stores one trained regression and the colour used to draw it.
type model struct {
	name      string
	quadratic bool
	coef      []float64
	col       color.Color
}

// loadRows reads the CSV, finds columns by header name, builds features, sorts by year.
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
		t := float64(year - baseYear)
		rows = append(rows, Row{Year: year, CO2: co2, TimeIndex: t, YearSq: t * t})
	}
	sort.Slice(rows, func(a, b int) bool { return rows[a].Year < rows[b].Year })
	return rows, nil
}

// splitRows puts years up to lastYear in train and later years in test (no shuffling).
func splitRows(rows []Row, lastYear int) (train, test []Row) {
	for _, r := range rows {
		if r.Year <= lastYear {
			train = append(train, r)
		} else {
			test = append(test, r)
		}
	}
	return train, test
}

// pick returns one number from every row.
func pick(rows []Row, get func(Row) float64) []float64 {
	out := make([]float64, len(rows))
	for i, r := range rows {
		out[i] = get(r)
	}
	return out
}

// fitOLS fits CO2 on Time_Index (linear) or Time_Index + Year_Sq (quadratic).
func fitOLS(train []Row, quadratic bool) ([]float64, error) {
	y := pick(train, func(r Row) float64 { return r.CO2 })
	t := pick(train, func(r Row) float64 { return r.TimeIndex })
	if !quadratic {
		alpha, beta := stat.LinearRegression(t, y, nil, false)
		return []float64{alpha, beta}, nil
	}
	n := len(train)
	X := mat.NewDense(n, 3, nil)
	for i, r := range train {
		X.Set(i, 0, 1)
		X.Set(i, 1, r.TimeIndex)
		X.Set(i, 2, r.YearSq)
	}
	Y := mat.NewDense(n, 1, y)
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

// rmse is the root-mean-squared error of a model on a set of rows.
func rmse(m model, rows []Row) float64 {
	var sse float64
	for _, r := range rows {
		e := r.CO2 - predict(m.coef, r)
		sse += e * e
	}
	return math.Sqrt(sse / float64(len(rows)))
}

// maxFloat returns the largest value found in the given slices.
func maxFloat(lists ...[]float64) float64 {
	best := math.Inf(-1)
	for _, l := range lists {
		for _, v := range l {
			if v > best {
				best = v
			}
		}
	}
	return best
}

// niceCeil rounds v up to the next multiple of step.
func niceCeil(v, step float64) float64 {
	return math.Ceil(v/step) * step
}

// series builds plot points (X = Year) from rows using the value function f.
func series(rows []Row, f func(Row) float64) plotter.XYs {
	xys := make(plotter.XYs, len(rows))
	for i, r := range rows {
		xys[i].X = float64(r.Year)
		xys[i].Y = f(r)
	}
	return xys
}

// yearTicks puts a small tick every 5 years and a labelled tick every 10 years.
func yearTicks(min, max float64) []plot.Tick {
	var ticks []plot.Tick
	for y := math.Ceil(min/5) * 5; y <= max; y += 5 {
		t := plot.Tick{Value: y}
		if int(y)%10 == 0 {
			t.Label = strconv.Itoa(int(y))
		}
		ticks = append(ticks, t)
	}
	return ticks
}

// valueTicks returns a ticker with a labelled tick every step and a small tick halfway.
func valueTicks(step float64) plot.TickerFunc {
	return func(min, max float64) []plot.Tick {
		var ticks []plot.Tick
		half := step / 2
		for v := math.Ceil(min/half) * half; v <= max; v += half {
			t := plot.Tick{Value: v}
			if math.Mod(math.Abs(v), step) < 1e-9 {
				t.Label = strconv.Itoa(int(math.Round(v)))
			}
			ticks = append(ticks, t)
		}
		return ticks
	}
}

// minFloat returns the smallest value in xs.
func minFloat(xs []float64) float64 {
	best := math.Inf(1)
	for _, v := range xs {
		if v < best {
			best = v
		}
	}
	return best
}

// newPlot creates a plot with a clean academic style (titles, fonts, light grid).
func newPlot(title, xLabel, yLabel string, yearAxis bool) *plot.Plot {
	p := plot.New()
	p.Title.Text = title
	p.X.Label.Text = xLabel
	p.Y.Label.Text = yLabel
	p.Title.TextStyle.Font.Size = 16
	p.X.Label.TextStyle.Font.Size = 13
	p.Y.Label.TextStyle.Font.Size = 13
	p.X.Tick.Label.Font.Size = 11
	p.Y.Tick.Label.Font.Size = 11
	p.Legend.TextStyle.Font.Size = 11
	p.Legend.Top = true
	p.Legend.Left = true
	if yearAxis {
		p.X.Tick.Marker = plot.TickerFunc(yearTicks)
	}
	g := plotter.NewGrid()
	g.Horizontal.Color = colGrid
	g.Vertical.Color = colGrid
	p.Add(g)
	return p
}

// finish fixes the axis ranges (must be called after everything is added to the plot).
func finish(p *plot.Plot, xMin, xMax, yMin, yMax float64) {
	p.X.Min, p.X.Max = xMin, xMax
	p.Y.Min, p.Y.Max = yMin, yMax
}

// addLine draws a line; a non-empty name also adds it to the legend.
func addLine(p *plot.Plot, xys plotter.XYs, col color.Color, width float64, dashed bool, name string) error {
	l, err := plotter.NewLine(xys)
	if err != nil {
		return err
	}
	l.LineStyle.Color = col
	l.LineStyle.Width = vg.Points(width)
	if dashed {
		l.LineStyle.Dashes = []vg.Length{vg.Points(6), vg.Points(4)}
	}
	p.Add(l)
	if name != "" {
		p.Legend.Add(name, l)
	}
	return nil
}

// addScatter draws markers; a non-empty name also adds them to the legend.
func addScatter(p *plot.Plot, xys plotter.XYs, col color.Color, shape draw.GlyphDrawer, radius float64, name string) error {
	s, err := plotter.NewScatter(xys)
	if err != nil {
		return err
	}
	s.GlyphStyle.Color = col
	s.GlyphStyle.Shape = shape
	s.GlyphStyle.Radius = vg.Points(radius)
	p.Add(s)
	if name != "" {
		p.Legend.Add(name, s)
	}
	return nil
}

// shadeTest draws a light grey band over the test years and returns it for the legend.
func shadeTest(p *plot.Plot, test []Row, yMin, yMax float64) (*plotter.Polygon, error) {
	x0 := float64(test[0].Year) - 0.5
	x1 := float64(test[len(test)-1].Year) + 0.5
	poly, err := plotter.NewPolygon(plotter.XYs{{X: x0, Y: yMin}, {X: x1, Y: yMin}, {X: x1, Y: yMax}, {X: x0, Y: yMax}})
	if err != nil {
		return nil, err
	}
	poly.Color = colShade
	poly.LineStyle.Width = 0
	p.Add(poly)
	return poly, nil
}

// savePNG renders the plot at high resolution and writes it to a PNG file.
func savePNG(p *plot.Plot, path string) error {
	c := vgimg.NewWith(vgimg.UseWH(imgWidth, imgHeight), vgimg.UseDPI(imgDPI))
	p.Draw(draw.New(c))
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	png := vgimg.PngCanvas{Canvas: c}
	if _, err := png.WriteTo(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// plotHistorical draws chart 1: the raw annual CO2 series.
func plotHistorical(all []Row, path string) error {
	title := fmt.Sprintf("Indonesia: Annual CO₂ Emissions, %d–%d", all[0].Year, all[len(all)-1].Year)
	p := newPlot(title, "Year", yLabelCO2, true)
	p.Y.Tick.Marker = valueTicks(100)
	xys := series(all, func(r Row) float64 { return r.CO2 })
	if err := addLine(p, xys, colQuad, 2.5, false, "Annual CO₂ emissions"); err != nil {
		return err
	}
	if err := addScatter(p, xys, colQuad, draw.CircleGlyph{}, 2.5, ""); err != nil {
		return err
	}
	yMax := niceCeil(maxFloat(pick(all, func(r Row) float64 { return r.CO2 })), 100)
	finish(p, float64(all[0].Year)-2, float64(all[len(all)-1].Year)+2, 0, yMax)
	return savePNG(p, path)
}

// plotActualVsModel draws charts 2 and 3: actual data against one model's predictions.
func plotActualVsModel(m model, all, train, test []Row, title, path string) error {
	xLabel := fmt.Sprintf("Year (model fitted on %d–%d only)", train[0].Year, train[len(train)-1].Year)
	p := newPlot(title, xLabel, yLabelCO2, true)
	p.Y.Tick.Marker = valueTicks(100)
	actual := pick(all, func(r Row) float64 { return r.CO2 })
	pred := pick(all, func(r Row) float64 { return predict(m.coef, r) })
	yMax := niceCeil(maxFloat(actual, pred), 100)
	yMin := math.Min(0, math.Floor(minFloat(pred)/50)*50) // shows negative predictions, if any
	poly, err := shadeTest(p, test, yMin, yMax)
	if err != nil {
		return err
	}
	actualXY := series(all, func(r Row) float64 { return r.CO2 })
	if err := addLine(p, actualXY, colActual, 2.5, false, "Actual emissions"); err != nil {
		return err
	}
	if err := addScatter(p, actualXY, colActual, draw.CircleGlyph{}, 2.2, ""); err != nil {
		return err
	}
	label := fmt.Sprintf("%s predictions (test RMSE %.1f Mt)", m.name, rmse(m, test))
	predXY := series(all, func(r Row) float64 { return predict(m.coef, r) })
	if err := addLine(p, predXY, m.col, 2.5, true, label); err != nil {
		return err
	}
	p.Legend.Add(fmt.Sprintf("Test period (%d–%d)", test[0].Year, test[len(test)-1].Year), poly)
	finish(p, float64(all[0].Year)-2, float64(all[len(all)-1].Year)+2, yMin, yMax)
	return savePNG(p, path)
}

// plotCurves draws chart 4: the Linear and Quadratic curves together, with actual data for reference.
func plotCurves(models []model, all, train, test []Row, path string) error {
	xLabel := fmt.Sprintf("Year (models fitted on %d–%d only; t = Year − %d)", train[0].Year, train[len(train)-1].Year, baseYear)
	p := newPlot("Linear vs Quadratic Regression Curves", xLabel, yLabelCO2, true)
	p.Y.Tick.Marker = valueTicks(100)
	lists := [][]float64{pick(all, func(r Row) float64 { return r.CO2 })}
	for _, m := range models {
		mm := m
		lists = append(lists, pick(all, func(r Row) float64 { return predict(mm.coef, r) }))
	}
	yMax := niceCeil(maxFloat(lists...), 100)
	yMin := 0.0
	for _, l := range lists[1:] {
		yMin = math.Min(yMin, math.Floor(minFloat(l)/50)*50)
	}
	poly, err := shadeTest(p, test, yMin, yMax)
	if err != nil {
		return err
	}
	if err := addScatter(p, series(all, func(r Row) float64 { return r.CO2 }), colFaint, draw.CircleGlyph{}, 2.6, "Actual emissions"); err != nil {
		return err
	}
	for _, m := range models {
		mm := m
		var eq string
		if mm.quadratic {
			eq = fmt.Sprintf("Quadratic: %.2f + %.2f t + %.3f t²", mm.coef[0], mm.coef[1], mm.coef[2])
		} else {
			eq = fmt.Sprintf("Linear: %.2f + %.2f t", mm.coef[0], mm.coef[1])
		}
		xys := series(all, func(r Row) float64 { return predict(mm.coef, r) })
		if err := addLine(p, xys, mm.col, 3, false, eq); err != nil {
			return err
		}
	}
	p.Legend.Add(fmt.Sprintf("Test period (%d–%d)", test[0].Year, test[len(test)-1].Year), poly)
	finish(p, float64(all[0].Year)-2, float64(all[len(all)-1].Year)+2, yMin, yMax)
	return savePNG(p, path)
}

// plotResiduals draws charts 5 and 6: residuals (actual - predicted) over time for one model.
func plotResiduals(m model, train, test []Row, lim float64, title, path string) error {
	p := newPlot(title, "Year", "Residual = actual − predicted (Mt CO₂e)", true)
	p.Y.Tick.Marker = valueTicks(100)
	poly, err := shadeTest(p, test, -lim, lim)
	if err != nil {
		return err
	}
	resid := func(r Row) float64 { return r.CO2 - predict(m.coef, r) }
	all := append(append([]Row{}, train...), test...)
	x0 := float64(all[0].Year) - 2
	x1 := float64(all[len(all)-1].Year) + 2
	if err := addLine(p, plotter.XYs{{X: x0, Y: 0}, {X: x1, Y: 0}}, colZero, 1.5, true, ""); err != nil {
		return err
	}
	if err := addLine(p, series(all, resid), colFaint, 1, false, ""); err != nil {
		return err
	}
	trainName := fmt.Sprintf("Train residuals (%d–%d)", train[0].Year, train[len(train)-1].Year)
	testName := fmt.Sprintf("Test residuals (%d–%d)", test[0].Year, test[len(test)-1].Year)
	if err := addScatter(p, series(train, resid), m.col, draw.CircleGlyph{}, 3, trainName); err != nil {
		return err
	}
	if err := addScatter(p, series(test, resid), m.col, draw.PyramidGlyph{}, 4.5, testName); err != nil {
		return err
	}
	p.Legend.Add(fmt.Sprintf("Test period (%d–%d)", test[0].Year, test[len(test)-1].Year), poly)
	finish(p, x0, x1, -lim, lim)
	return savePNG(p, path)
}

// plotResidualsVsFitted draws chart 7: residuals against fitted values for both models.
func plotResidualsVsFitted(models []model, train, test []Row, lim float64, path string) error {
	p := newPlot("Residuals vs Fitted Values", "Fitted value (Mt CO₂e)", "Residual = actual − predicted (Mt CO₂e)", false)
	p.X.Tick.Marker = valueTicks(100)
	p.Y.Tick.Marker = valueTicks(100)
	all := append(append([]Row{}, train...), test...)
	fMin, fMax := math.Inf(1), math.Inf(-1)
	for _, m := range models {
		for _, r := range all {
			f := predict(m.coef, r)
			fMin = math.Min(fMin, f)
			fMax = math.Max(fMax, f)
		}
	}
	x0, x1 := math.Floor(fMin/50)*50-25, niceCeil(fMax, 50)+25
	if err := addLine(p, plotter.XYs{{X: x0, Y: 0}, {X: x1, Y: 0}}, colZero, 1.5, true, ""); err != nil {
		return err
	}
	for _, m := range models {
		mm := m
		pts := func(rows []Row) plotter.XYs {
			xys := make(plotter.XYs, len(rows))
			for i, r := range rows {
				xys[i].X = predict(mm.coef, r)
				xys[i].Y = r.CO2 - predict(mm.coef, r)
			}
			return xys
		}
		if err := addScatter(p, pts(train), mm.col, draw.CircleGlyph{}, 3, mm.name+" – train"); err != nil {
			return err
		}
		if err := addScatter(p, pts(test), mm.col, draw.PyramidGlyph{}, 4.5, mm.name+" – test"); err != nil {
			return err
		}
	}
	finish(p, x0, x1, -lim, lim)
	return savePNG(p, path)
}

// fail prints an error and stops the program.
func fail(step string, err error) {
	fmt.Fprintf(os.Stderr, "error (%s): %v\n", step, err)
	os.Exit(1)
}

// main loads data, refits both models and writes all figures.
func main() {
	csvPath := "NGDS_CO2_Indonesia.csv"
	if len(os.Args) > 1 {
		csvPath = os.Args[1]
	}
	all, err := loadRows(csvPath)
	if err != nil {
		fail("load "+csvPath, err)
	}
	train, test := splitRows(all, lastTrainYear)
	if len(train) < 3 || len(test) == 0 {
		fail("split", fmt.Errorf("train or test set is too small"))
	}

	models := []model{
		{name: "Linear", col: colLinear},
		{name: "Quadratic", quadratic: true, col: colQuad},
	}
	for i := range models {
		models[i].coef, err = fitOLS(train, models[i].quadratic)
		if err != nil {
			fail("fit "+models[i].name, err)
		}
		fmt.Printf("%-9s coefficients %v   test RMSE %.3f\n", models[i].name, models[i].coef, rmse(models[i], test))
	}

	// One common y-limit so the two residual charts can be compared fairly.
	maxAbs := 0.0
	for _, m := range models {
		for _, r := range all {
			maxAbs = math.Max(maxAbs, math.Abs(r.CO2-predict(m.coef, r)))
		}
	}
	lim := niceCeil(maxAbs*1.1, 50)

	outDir := "figures"
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("create "+outDir, err)
	}
	out := func(name string) string { return filepath.Join(outDir, name) }

	steps := []struct {
		file string
		run  func() error
	}{
		{"01_historical_co2.png", func() error { return plotHistorical(all, out("01_historical_co2.png")) }},
		{"02_actual_vs_linear.png", func() error {
			return plotActualVsModel(models[0], all, train, test, "Actual Emissions vs Linear Regression", out("02_actual_vs_linear.png"))
		}},
		{"03_actual_vs_quadratic.png", func() error {
			return plotActualVsModel(models[1], all, train, test, "Actual Emissions vs Quadratic Regression", out("03_actual_vs_quadratic.png"))
		}},
		{"04_linear_vs_quadratic.png", func() error { return plotCurves(models, all, train, test, out("04_linear_vs_quadratic.png")) }},
		{"05_residuals_linear.png", func() error {
			return plotResiduals(models[0], train, test, lim, "Linear Regression: Residuals Over Time", out("05_residuals_linear.png"))
		}},
		{"06_residuals_quadratic.png", func() error {
			return plotResiduals(models[1], train, test, lim, "Quadratic Regression: Residuals Over Time", out("06_residuals_quadratic.png"))
		}},
		{"07_residuals_vs_fitted.png", func() error {
			return plotResidualsVsFitted(models, train, test, lim, out("07_residuals_vs_fitted.png"))
		}},
	}
	for _, s := range steps {
		if err := s.run(); err != nil {
			fail(s.file, err)
		}
		fmt.Println("saved", out(s.file))
	}
}