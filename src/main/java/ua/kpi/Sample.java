package ua.kpi;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.io.Serializable;
import java.util.List;

import static org.apache.spark.sql.functions.col;

public class Sample extends ApplicationFrame {
  public Sample(String title) {
    super(title);
  }

  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder()
        .appName("Linear Regression")
        .master("local[*]")
        .getOrCreate();
    JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

    Dataset<Row> data = spark.read().format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load("src/main/resources/dataset.csv");
    data = data.na().drop();
    data = data.dropDuplicates();

    data = data
        .withColumn("message_size", col("message_size").cast(DataTypes.DoubleType))
        .withColumn("delivery_time", col("delivery_time").cast(DataTypes.DoubleType));

    JavaRDD<DataPoint> dataPoints = data.javaRDD().map(row -> new DataPoint(row.getDouble(0), row.getDouble(1)));

    SimpleRegression regression = new SimpleRegression();
    dataPoints.collect().forEach(dp -> regression.addData(dp.getX(), dp.getY()));

    double intercept = regression.getIntercept();
    double slope = regression.getSlope();
    double rSquared = regression.getRSquare();
    double correlation = regression.getR();
    double slopeStdErr = regression.getSlopeStdErr();
    double interceptStdErr = regression.getInterceptStdErr();
    double meanSquareError = regression.getMeanSquareError();

    System.out.println("Intercept: " + intercept);
    System.out.println("Slope: " + slope);
    System.out.println("R-Squared: " + rSquared);
    System.out.println("Correlation: " + correlation);
    System.out.println("Slope Std Error: " + slopeStdErr);
    System.out.println("Intercept Std Error: " + interceptStdErr);
    System.out.println("Mean Square Error: " + meanSquareError);

    checkStatisticalSignificance(rSquared, correlation, dataPoints.collect().size());
    double confidenceLevel = 0.95;
    double[] confidenceInterval = calculateConfidenceInterval(dataPoints.collect(), regression, confidenceLevel);
    System.out.println("Prediction interval for mean value with confidence level "
        + confidenceLevel
        + ": ["
        + confidenceInterval[0]
        + ", "
        + confidenceInterval[1]
        + "]");

    Sample sample = new Sample("Linear Regression");
    sample.visualize(dataPoints.collect(), intercept, slope, confidenceInterval);
    jsc.close();
  }

  private static void checkStatisticalSignificance(double rSquared, double correlation, int sampleSize) {
    double fStatistic = (rSquared / (1 - rSquared)) * ((sampleSize - 2) / 1);
    System.out.println("F-Statistic for R-Squared: " + fStatistic);
    double tStatistic = correlation * Math.sqrt((sampleSize - 2) / (1 - correlation * correlation));
    System.out.println("t-Statistic for Correlation: " + tStatistic);
    int degreesOfFreedom = sampleSize - 2;

    TDistribution tDist = new TDistribution(degreesOfFreedom);
    double pValueF = 1 - tDist.cumulativeProbability(fStatistic);
    double pValueT = 2 * (1 - tDist.cumulativeProbability(Math.abs(tStatistic)));

    System.out.println("p-Value for F-Statistic: " + pValueF);
    System.out.println("p-Value for t-Statistic: " + pValueT);

    double alpha = 0.05;

    if (pValueF < alpha) {
      System.out.println("R-Squared is statistically significant.");
    } else {
      System.out.println("R-Squared is not statistically significant.");
    }

    if (pValueT < alpha) {
      System.out.println("Correlation is statistically significant.");
    } else {
      System.out.println("Correlation is not statistically significant.");
    }
  }

  private static double[] calculateConfidenceInterval(List<DataPoint> dataPoints, SimpleRegression regression, double confidenceLevel) {
    int n = dataPoints.size();
    double meanX = dataPoints.stream().mapToDouble(DataPoint::getX).average().orElse(0);
    double sumSquaredDeviations = dataPoints.stream().mapToDouble(dp -> Math.pow(dp.getX() - meanX, 2)).sum();
    double meanY = regression.predict(meanX);
    System.out.println("Predicted mean value: " + meanY);

    double standardError = Math.sqrt(regression.getMeanSquareError() * (1.0 / n + (Math.pow(meanX, 2) / sumSquaredDeviations)));

    TDistribution tDistribution = new TDistribution(n - 2);
    double tValue = tDistribution.inverseCumulativeProbability(1.0 - (1.0 - confidenceLevel) / 2.0);

    double marginOfError = tValue * standardError;
    return new double[]{meanY - marginOfError, meanY + marginOfError};
  }

  private void visualize(List<DataPoint> dataPoints, double intercept, double slope, double[] confidenceInterval) {
    XYSeries series = new XYSeries("Data Points");
    XYSeries regressionLine = new XYSeries("Regression Line");
    XYSeries confidenceIntervalLine1 = new XYSeries("Lower Confidence Interval");
    XYSeries confidenceIntervalLine2 = new XYSeries("Upper Confidence Interval");

    for (DataPoint dp : dataPoints) {
      series.add(dp.getX(), dp.getY());
    }

    double minX = dataPoints.stream().mapToDouble(DataPoint::getX).min().orElse(0);
    double maxX = dataPoints.stream().mapToDouble(DataPoint::getX).max().orElse(0);

    regressionLine.add(minX, intercept + slope * minX);
    regressionLine.add(maxX, intercept + slope * maxX);

    confidenceIntervalLine1.add(minX, confidenceInterval[0]);
    confidenceIntervalLine1.add(maxX, confidenceInterval[0]);
    confidenceIntervalLine2.add(minX, confidenceInterval[1]);
    confidenceIntervalLine2.add(maxX, confidenceInterval[1]);

    XYSeriesCollection dataset = new XYSeriesCollection();
    dataset.addSeries(series);
    dataset.addSeries(regressionLine);
    dataset.addSeries(confidenceIntervalLine1);
    dataset.addSeries(confidenceIntervalLine2);

    JFreeChart chart = ChartFactory.createScatterPlot(
        "Linear Regression",
        "Message Size",
        "Delivery Time",
        dataset,
        PlotOrientation.VERTICAL,
        true,
        true,
        false
    );

    XYPlot plot = (XYPlot) chart.getPlot();
    XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
    renderer.setSeriesLinesVisible(0, false);
    renderer.setSeriesShapesVisible(0, true);
    renderer.setSeriesLinesVisible(1, true);
    renderer.setSeriesShapesVisible(1, false);
    renderer.setSeriesPaint(2, Color.RED);
    renderer.setSeriesLinesVisible(2, true);
    renderer.setSeriesShapesVisible(2, false);
    renderer.setSeriesPaint(3, Color.RED);
    renderer.setSeriesLinesVisible(3, true);
    renderer.setSeriesShapesVisible(3, false);

    plot.setRenderer(renderer);

    ChartPanel chartPanel = new ChartPanel(chart);
    chartPanel.setPreferredSize(new Dimension(800, 600));
    setContentPane(chartPanel);

    pack();
    setVisible(true);
    setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
  }

  static class DataPoint implements Serializable {
    private final double x;
    private final double y;

    public DataPoint(double x, double y) {
      this.x = x;
      this.y = y;
    }

    public double getX() {
      return x;
    }

    public double getY() {
      return y;
    }
  }
}
