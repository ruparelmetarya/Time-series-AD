

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.FileWriter;
import java.util.Arrays;

public class DateConverter {
/**
* @param args
* @throws ParseException
*/
  public static void main(String[] args) throws Exception {
    SimpleDateFormat sdf = new SimpleDateFormat("M/d/y h:m");
    sdf.setTimeZone(TimeZone.getTimeZone("UTC"));
    String csvFile = "/Users/mruparel/warden-analysis-workspace/scikit/data (2).csv";
    BufferedReader br = null;
    String line = "";
    String cvsSplitBy = ",";
    List<Long> timestamp_start = new ArrayList<Long>();
    List<Long> timestamp_stop = new ArrayList<Long>();
    List<String> pod = new ArrayList<String>();
    List<Long> sampleCount = new ArrayList<Long>();
    List<Long> thread_count_1 = new ArrayList<Long>();
    List<Long> thread_count_2 = new ArrayList<Long>();
    List<Long> thread_count_3 = new ArrayList<Long>();
    List<Long> thread_count_4 = new ArrayList<Long>();
    try {
      br = new BufferedReader(new FileReader(csvFile));
      while ((line = br.readLine()) != null) {
        String[] samples = line.split(cvsSplitBy);
        timestamp_start.add(Long.parseLong(samples[0]));
        timestamp_stop.add(Long.parseLong(samples[1]));
        pod.add(samples[2]);
        sampleCount.add(Long.parseLong(samples[3]));
        thread_count_1.add(Long.parseLong(samples[4]));
        thread_count_2.add(Long.parseLong(samples[5]));
        thread_count_3.add(Long.parseLong(samples[6]));
        thread_count_4.add(Long.parseLong(samples[7]));
      }
    }
    catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
        e.printStackTrace();
    } finally {
        if (br != null) {
          try {
            br.close();
          } catch (IOException e) {
              e.printStackTrace();
            }
        }
    }
    String CsvFile = "/Users/mruparel/warden-analysis-workspace/scikit/anomaly.csv";
    FileWriter writer = new FileWriter(CsvFile);
    for(int i = 0 ; i < timestamp_start.size() ; ++i) {
      long timeMillisStart = timestamp_start.get(i);
      long timeMillisStop = timestamp_stop.get(i);
      String Pod = pod.get(i);
      String SampleCount = String.valueOf(sampleCount.get(i));
      String Thread_count_1 = String.valueOf(thread_count_1.get(i));
      String Thread_count_2 = String.valueOf(thread_count_2.get(i));
      String Thread_count_3 = String.valueOf(thread_count_3.get(i));
      String Thread_count_4 = String.valueOf(thread_count_4.get(i));
    // long timeMillisSplunk = 1491027055046L;
      Date wardenDateStart = new Date(timeMillisStart);
      Date wardenDateStop = new Date(timeMillisStop);
    // Date splunkDate1 = new Date(timeMillisSplunk);
      System.out.println(sdf.format(wardenDateStart).toString());
      CSVUtils.writeLine(writer, Arrays.asList(sdf.format(wardenDateStart).toString(),SampleCount));
    // System.out.println("Splunk Date:"+sdf.format(splunkDate1).toString());
    // System.out.println("Diff Millis"+Math.abs(timeMillisSplunk-timeMillisWarden));
    }
    writer.flush();
    writer.close();
  }
}
