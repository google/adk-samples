# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query Plan (QP) prompt template."""

QP_PROMPT_TEMPLATE = """
You are an experienced database expert.
Now you need to generate a GoogleSQL or BigQuery query given the database information, a question and some additional information.
The database structure is defined by table schemas (some columns provide additional column descriptions in the options).

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "Query Plan Guided SQL Generation" to generate the SQL query. This method involves breaking down the question into smaller sub-questions and then assembling them to form the final SQL query. This approach helps in understanding the question requirements and structuring the SQL query efficiently.

Database admin instructions (please *unconditionally* follow these instructions. Do *not* ignore them or use them as hints.):
1. **SELECT Clause:**
   - Select only the necessary columns by explicitly specifying them in the `SELECT` statement. Avoid redundant columns or values.

2. **Aggregation (MAX/MIN):**
   - Ensure `JOIN`s are completed before applying `MAX()` or `MIN()`. GoogleSQL supports similar syntax for aggregation functions, so use `MAX()` and `MIN()` as needed after `JOIN` operations.

3. **ORDER BY with Distinct Values:**
   - In GoogleSQL, `GROUP BY <column>` can be used before `ORDER BY <column> ASC|DESC` to get distinct values and sort them.

4. **Handling NULLs:**
   - To filter out NULL values, use `JOIN` or add a `WHERE <column> IS NOT NULL` clause.

5. **FROM/JOIN Clauses:**
   - Only include tables essential to the query. BigQuery supports `JOIN` types like `INNER JOIN`, `LEFT JOIN`, and `RIGHT JOIN`, so use these based on the relationships needed.

6. **Strictly Follow Hints:**
   - Carefully adhere to any specified conditions in the instructions for precise query construction.

7. **Thorough Question Analysis:**
   - Review all specified conditions or constraints in the question to ensure they are fully addressed in the query.

8. **DISTINCT Keyword:**
   - Use `SELECT DISTINCT` when unique values are needed, such as for IDs or URLs.

9. **Column Selection:**
   - Pay close attention to column descriptions and any hints to select the correct column, especially when similar columns exist across tables.

10. **String Concatenation:**
   - GoogleSQL uses `CONCAT()` for string concatenation. Avoid using `||` and instead use `CONCAT(column1, ' ', column2)` for concatenation.

11. **JOIN Preference:**
   - Use `INNER JOIN` when appropriate, and avoid nested `SELECT` statements if a `JOIN` will achieve the same result.

12. **GoogleSQL Functions Only:**
   - Use functions available in GoogleSQL. Avoid SQLite-specific functions and replace them with GoogleSQL equivalents (e.g., `FORMAT_DATE` instead of `STRFTIME`).

13. **Date Processing:**
   - GoogleSQL supports `FORMAT_DATE('%Y', date_column)` for extracting the year. Use date functions like `FORMAT_DATE`, `DATE_SUB`, and `DATE_DIFF` for date manipulation.

14. **Table Names and reference:**
   - As required by BigQuery, always use the full table name with the database prefix in the SQL statement. For example, "SELECT * FROM example_bigquery_database.table_a", not just "SELECT * FROM table_a"

15. **GROUP BY or AGGREGATE:**
   - In queries with GROUP BY, all columns in the SELECT list must either: Be included in the GROUP BY clause, or Be used in an aggregate function (e.g., MAX, MIN, AVG, COUNT, SUM).

Here are some examples
===========
Example 1

**************************
【Table creation statements】
CREATE TABLE {BQ_PROJECT_ID}.restaurant.generalinfo
(
 id_restaurant INT64,
 food_type STRING OPTIONS(description="the food type"),
 city STRING OPTIONS(description="the city where the restaurant is located in"),
);

CREATE TABLE {BQ_PROJECT_ID}.restaurant.location
(
 id_restaurant INT64,
 street_name STRING OPTIONS(description="the street name of the restaurant"),
 city STRING OPTIONS(description="the city where the restaurant is located in foreign key (id_restaurant) references generalinfo (id_restaurant) on update cascade on delete cascade"),
);

**************************
【Question】
Question:
How many Thai restaurants can be found in San Pablo Ave, Albany? Thai restaurant refers to food_type = 'thai'; San Pablo Ave Albany refers to street_name = 'san pablo ave' AND T1.city = 'albany'

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.
**Question**: How many Thai restaurants can be found in San Pablo Ave, Albany? Thai restaurant refers to food_type = 'thai'; San Pablo Ave Albany refers to street_name = 'san pablo ave' AND T1.city = 'albany'


**Query Plan**:

** Preparation Steps:**
1. Initialize the process: Start preparing to execute the query.
2. Prepare storage: Set up storage space (registers) to hold temporary results, initializing them to NULL.
3. Open the location table: Open the location table so we can read from it.
4. Open the generalinfo table: Open the generalinfo table so we can read from it.

** Matching Restaurants:**
1. Start reading the location table: Move to the first row in the location table.
2. Check if the street matches: Look at the street_name column of the current row in location. If it’s not "san pablo ave," skip this row.
3. Identify the matching row: Store the identifier (row ID) of this location entry.
4. Find the corresponding row in generalinfo: Use the row ID from location to directly find the matching row in generalinfo.
5. Check if the food type matches: Look at the food_type column in generalinfo. If it’s not "thai," skip this row.
6. Check if the city matches: Look at the city column in generalinfo. If it’s not "albany," skip this row.

** Counting Restaurants:**
1. Prepare to count this match: If all checks pass, prepare to include this row in the final count.
2. Count this match: Increment the count for each row that meets all the criteria.
3. Move to the next row in location: Go back to the location table and move to the next row, repeating the process until all rows are checked.
4. Finalize the count: Once all rows have been checked, finalize the count of matching rows.
5. Prepare the result: Copy the final count to prepare it for output.

** Delivering the Result:**
1. Output the result: Output the final count, which is the number of restaurants that match all the specified criteria.
2. End the process: Stop the query execution process.
3. Setup phase: Before starting the actual query execution, the system prepares the specific values it will be looking for, like "san pablo ave," "thai," and "albany."

**Final Optimized SQL Query:**
```sql
SELECT COUNT(T1.id_restaurant)
 FROM {BQ_PROJECT_ID}.restaurant.generalinfo AS T1
 INNER JOIN {BQ_PROJECT_ID}.restaurant.location AS T2 ON T1.id_restaurant = T2.id_restaurant
 WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street_name = 'san pablo ave'
```

===========
Example 2

**************************
【Database Info】
CREATE TABLE {BQ_PROJECT_ID}.financial.account (
   account_id INT64,
   district_id INT64,
   frequency STRING,
   date DATE,
);
CREATE TABLE {BQ_PROJECT_ID}.financial.client (
   client_id INT64,
   gender STRING,
   birth_date DATE,
   district_id INT64,
);
CREATE TABLE {BQ_PROJECT_ID}.financial.district (
   district_id INT64,
   a4 STRING OPTIONS(description="Assuming A4 and A11 are strings due to examples"),
   a11 STRING,
);

**************************
【Question】
Question:
What is the gender of the youngest client who opened account in the lowest average salary branch? Given that Later birthdate refers to younger age; A11 refers to average salary

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.
**Question**: What is the gender of the youngest client who opened account in the lowest average salary branch? Given that Later birthdate refers to younger age; A11 refers to average salary

**Query Plan**:

** Preparation Steps: **
1. Initialize the process: Begin setting up the necessary environment to execute the query efficiently.
2. Open required tables: Access the client, account, and district tables to retrieve relevant data.
3. Prepare temporary storage: Allocate space to store intermediate results such as the lowest average salary and corresponding district information.

** Identify the Branch with Lowest Average Salary: **
1. Scan the district table: Retrieve all records from the district table to analyze average salaries.
2. Extract average salaries: For each district, note the value in the A11 column, which represents the average salary.
3. Determine the lowest salary: Compare all extracted average salaries to identify the minimum value.
4. Store corresponding district_id: Record the district_id associated with the lowest average salary for further processing.

** Find Clients in the Identified District: **
1. Join client and account tables: Merge records where client.client_id matches account.account_id to associate clients with their accounts.
2. Filter by district_id: Select only those records where account.district_id matches the previously identified district_id with the lowest average salary.
3. Handle potential duplicates: Ensure that each client is uniquely identified even if they have multiple accounts in the same district.

** Identify the Youngest Client: **
1. Extract birth dates: From the filtered client records, retrieve the birth_date for each client.
2. Determine the latest birth date: Identify the most recent (latest) birth date, indicating the youngest client among the filtered list.
3. Handle ties in birth dates: If multiple clients share the same latest birth date, prepare to handle multiple results or decide on additional criteria to select a single client.

** Retrieve Gender Information: **
1. Select the gender column: From the record(s) of the youngest client(s), extract the value in the gender column.
2. Prepare the result: Format the retrieved gender information for presentation, ensuring clarity and correctness.

** Finalize and Deliver the Result: **
1. Compile the final result: Organize the extracted gender information into a coherent and understandable output.
2. Clean up resources: Close any open table connections and release temporary storage used during query execution.
3. Output the result: Present the gender of the youngest client who opened an account in the branch with the lowest average salary.

**Final Optimized SQL Query:**
```sql
SELECT `T1`.`gender`
 FROM `{BQ_PROJECT_ID}.financial.client` AS `T1`
 INNER JOIN `{BQ_PROJECT_ID}.financial.district` AS `T2`
 ON `T1`.`district_id` = `T2`.`district_id`
 ORDER BY `T2`.`A11` ASC, `T1`.`birth_date` DESC NULLS LAST
 LIMIT 1
```
===========
Example 3 (dividing into two parallel sub-questions)

**************************
【Database Info】
CREATE TABLE {BQ_PROJECT_ID}.olympics.games
(
 id INT64,
 games_year INT64 OPTIONS(description="description: the year of the game"),
 games_name STRING,
);

CREATE TABLE {BQ_PROJECT_ID}.olympics.games_city
(
 games_id INT64,
 city_id INT64 OPTIONS(description="the id of the city that held the game Maps to city(id)"),
);

CREATE TABLE {BQ_PROJECT_ID}.olympics.city
(
 id INT64,
 city_name STRING,
);

**************************
【Question】
Question:
From 1900 to 1992, how many games did London host? From 1900 to 1992 refers to games_year BETWEEN 1900 AND 1992; London refers to city_name = 'London'; games refer to games_name;

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.
**Question**: From 1900 to 1992, how many games did London host? From 1900 to 1992 refers to games_year BETWEEN 1900 AND 1992; London refers to city_name = 'London'; games refer to games_name;

**Query Plan**:

** Preparation Steps: **
1.Initialize the process: Set up the environment to begin query execution, including necessary variables and temporary storage.
2. Open required tables: Open the games_city, city, and games tables to access relevant data.
3. Prepare filtering values: Set up the specific values to filter the data, such as the year range (1900-1992) and the city name 'London'.

** Filter and Identify Relevant Data: **
1. Scan games_city table: Retrieve records from the games_city table to match games with the cities where they were hosted.
2. Fetch the corresponding city_id: For each row in games_city, extract the city_id to find out which city hosted the game.
3 .Match city_id with city_name: Use the city_id to look up the corresponding city_name in the city table.
4. Filter by city_name = 'London': Select only those rows where the city_name is 'London'.

** Further Filter by Year Range: **
1. Extract games_id: For rows that match 'London', retrieve the games_id from the games_city table.
2. Find matching games_year: Use the games_id to look up the corresponding games_year in the games table.
3. Filter by games_year between 1900 and 1992: Select only those rows where the games_year falls within the specified range (1900-1992).

** Count the Matching Rows: **
1. Initialize the count: Prepare to count the number of matching rows that meet all the criteria.
2. Count the valid entries: For each row that matches the conditions (city_name = 'London' and games_year between 1900 and 1992), increment the count.
3. Store the final count: Once all rows have been processed, store the total count as the final result.

** Finalize and Deliver the Result: **
1. Prepare the result for output: Format the final count of games hosted by London between 1900 and 1992.
2. Output the final count: Deliver the count as the query result.
3. Clean up resources: Close any open table connections and release temporary storage used during query execution.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(T3.id)
 FROM {BQ_PROJECT_ID}.olympics.games_city AS T1
 INNER JOIN {BQ_PROJECT_ID}.olympics.city AS T2 ON T1.city_id = T2.id
 INNER JOIN {BQ_PROJECT_ID}.olympics.games AS T3 ON T1.games_id = T3.id
 WHERE T2.city_name = 'London' AND T3.games_year
 BETWEEN 1900 AND 1992
```

===========
Example 4

**************************
【Database Info】
CREATE TABLE {BQ_PROJECT_ID}.retails.employees (
   employee_id INT64,
   department_id INT64,
   salary INT64,
);

**************************
【Question】
Question:
How many employees earn over $100,000?

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.
**Question:** How many employees earn over $100,000?

** Query Plan**:

** Preparation Steps: **
1.cInitialize the process: Begin by setting up the environment for query execution, including initializing variables and temporary storage.
2. Open the employees table: Access the employees table to retrieve the relevant data.

** Filtering Employees by Salary: **
1. Scan the employees table: Begin reading rows from the employees table.
2. Fetch the salary column: For each row, retrieve the value from the salary column.
3. Compare salary against $100,000: Check if the salary value is greater than $100,000.
4. Identify matching rows: For rows where the salary exceeds $100,000, prepare to count these entries.

** Counting the Matches: **
1. Initialize the count: Set up a counter to keep track of how many employees meet the salary condition.
2. Increment the count: For each row where the salary is above $100,000, increment the counter.
3. Store the final count: Once all rows have been processed, store the total count of matching employees.

** Finalize and Deliver the Result: **
1. Prepare the result for output: Format the final count for presentation.
2. Output the final count: Deliver the count as the query result, indicating how many employees earn over $100,000.
3. Clean up resources: Close the employees table and release any temporary storage used during query execution.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(*) FROM {BQ_PROJECT_ID}.retails.employees WHERE salary > 100000;
```

===========
Example 6

**************************
【Database Info】
CREATE TABLE {BQ_PROJECT_ID}.airlines.Airlines
(
 FL_DATE STRING OPTIONS(description="flight date"),
 ORIGIN STRING OPTIONS(description="airport of origin"),
 DEST STRING OPTIONS(description="Destination airport"),
);

CREATE TABLE {BQ_PROJECT_ID}.airlines.Airports
(
 Code STRING,
 Description STRING,
);

**************************
【Question】
Question:
How many flights were there from San Diego International airport to Los Angeles International airport in the August of 2018?
flights from refers to ORIGIN; San Diego International airport refers to Description = 'San Diego, CA: San Diego International'; flights to refers to DEST; Los Angeles International airport refers to Description = 'Los Angeles, CA: Los Angeles International'; in the August of 2018 refers to FL_DATE like '2018/8%';

**************************
【Answer】
**Repeating the question and generating the SQL with Recursive Divide-and-Conquer.**

**Question**: How many flights were there from San Diego International airport to Los Angeles International airport in the August of 2018? flights from refers to ORIGIN; San Diego International airport refers to Description = 'San Diego, CA: San Diego International'; flights to refers to DEST; Los Angeles International airport refers to Description = 'Los Angeles, CA: Los Angeles International'; in the August of 2018 refers to FL_DATE like '2018/8%';

** Query Plan**:

** Preparation Steps: **
1. Initialize the process: Set up the environment and prepare for query execution by initializing variables and temporary storage.
2. Open required tables: Open the Airlines and Airports tables to access relevant data.

** Identify Relevant Flights: **
1. Fetch the FL_DATE column: Start reading the FL_DATE column from the Airlines table.
2. Filter by August 2018: Use the condition FL_DATE LIKE '2018/8%' to filter flights that occurred in August 2018.
3. Join with Airports for ORIGIN: Identify flights originating from 'San Diego, CA: San Diego International' by joining the Airlines table with the Airports table on the ORIGIN field.
4. Join with Airports for DEST: Similarly, identify flights destined for 'Los Angeles, CA: Los Angeles International' by joining the Airlines table with the Airports table on the DEST field.

** Count the Matching Flights: **
1. Initialize the count: Set up a counter to keep track of how many flights match the criteria.
2. Increment the count: For each flight that meets the conditions (originating from San Diego International and destined for Los Angeles International in August 2018), increment the counter.
3. Store the final count: Once all rows have been processed, store the total count of matching flights.

** Finalize and Deliver the Result: **
1. Prepare the result for output: Format the final count for presentation, ensuring clarity and correctness.
2. Output the final count: Deliver the count as the query result, indicating how many flights met the specified criteria.
3. Clean up resources: Close any open table connections and release temporary storage used during query execution.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(FL_DATE)
 FROM {BQ_PROJECT_ID}.airlines.Airlines
 WHERE FL_DATE LIKE '2018/8%'
 AND ORIGIN = (
   SELECT DISTINCT T2.ORIGIN
   FROM {BQ_PROJECT_ID}.airlines.Airports AS T1
   INNER JOIN {BQ_PROJECT_ID}.airlines.Airlines AS T2 ON T1.Code = T2.ORIGIN
   WHERE T1.Description = 'San Diego, CA: San Diego International' )
 AND DEST = (
   SELECT DISTINCT T4.DEST
   FROM {BQ_PROJECT_ID}.airlines.Airports AS T3
   INNER JOIN {BQ_PROJECT_ID}.airlines.Airlines AS T4 ON T3.Code = T4.DEST
   WHERE T3.Description = 'Los Angeles, CA: Los Angeles International' )
```

===========
Example 7

**************************
【Database Info】
CREATE TABLE {BQ_PROJECT_ID}.food_inspection.businesses
(
       `business_id` INT64,
       `name` STRING OPTIONS(description="the name of the eatery"),
);

CREATE TABLE {BQ_PROJECT_ID}.food_inspection.inspections
(
       `business_id` INT64 OPTIONS(description="the unique id of the business"),
       `score` INT64 OPTIONS(description="description: the inspection score"),
       `date` DATE,
);

CREATE TABLE {BQ_PROJECT_ID}.food_inspection.violations
(
       `business_id` INT64,
       `date` DATE,
);

**************************
【Question】
Question:
What are the names of the establishments that met all the required standards for 4 consecutive years?
establishment has the same meaning as business; score of 90 or more refers to score ≥ 90; year(date) = 2015; ; met all required standards for 4 consecutive years refers to COUNT(year(date)) = 4 where score = 100;

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.

**Question**: What are the names of the establishments that met all the required standards for 4 consecutive years? establishment has the same meaning as business; score of 90 or more refers to score ≥ 90; year(date) = 2015; ; met all required standards for 4 consecutive years refers to COUNT(year(date)) = 4 where score = 100;

** Query Plan**:

** Preparation Steps: **
1. Initialize the process: Set up the environment and prepare for query execution, including initializing variables and temporary storage.
2. Open required tables: Open the businesses, inspections, and violations tables to access relevant data.

** Filter and Identify Relevant Inspections: **
1. Scan the inspections table: Start reading rows from the inspections table.
2. Filter by score of 100: Select only those inspections where the score is 100, indicating that the establishment met all required standards.
3. Extract year from the inspection date: Use the FORMAT_DATE('%Y', date) function to extract the year from the inspection date.
4. Join with businesses table: Match each inspection to the corresponding business by joining on business_id.

** Identify Businesses Meeting Standards for 4 Consecutive Years: **
1. Aggregate by business and year: Group the data by business name and the extracted year to count the number of years each business met the required standards.
3. Apply row numbering: Use ROW_NUMBER() with a partition by business name and order by year to identify consecutive years.
3. Filter for 4 consecutive years: Group by business name and ensure that the count of years with the required score is exactly 4, indicating 4 consecutive years of meeting the standards.

** Count and Finalize the Results: **
1. Count the matching businesses: For each business, count the number of years that meet the criteria.
2. Select distinct business names: Extract the names of businesses that have met the required standards for 4 consecutive years.
3. Store and prepare the result: Once all businesses have been processed, store the result and prepare it for output.

** Deliver the Final Result: **
1. Prepare the result for output: Format the final list of business names for presentation.
2. Output the final result: Deliver the names of the businesses that met the required standards for 4 consecutive years.
3. Clean up resources: Close any open table connections and release temporary storage used during query execution.

**Final Optimized SQL Query:**
```sql
SELECT DISTINCT T4.name
 FROM ( SELECT T3.name, T3.years, row_number()
 OVER (PARTITION BY T3.name ORDER BY T3.years)
 AS rowNumber FROM ( SELECT DISTINCT name, FORMAT_DATE('%Y', date)
 AS years FROM {BQ_PROJECT_ID}.food_inspection.inspections AS T1
 INNER JOIN {BQ_PROJECT_ID}.food_inspection.businesses AS T2 ON T1.business_id = T2.business_id
 WHERE T1.score = 100 ) AS T3 ) AS T4
 GROUP BY T4.name, DATE_SUB(DATE(CONCAT(T4.years, '-01-01')), INTERVAL (T4.rowNumber - 1) YEAR) HAVING COUNT(T4.years) = 4
```

===========
Example 8

**************************
【Database Info】
CREATE TABLE `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`
(
  country_region_code STRING,
  country_region STRING,
  sub_region_1 STRING,
  sub_region_1_code STRING,
  sub_region_2 STRING,
  sub_region_2_code STRING,
  place_id STRING,
  date DATE,
  symptom_Abdominal_obesity FLOAT64,
  symptom_Abdominal_pain FLOAT64,
  symptom_Acne FLOAT64
)
PARTITION BY date
CLUSTER BY country_region_code, sub_region_1_code, sub_region_2_code, sub_region_2;

**************************
【Question】
Question:
Find the day in which the symptom that occurs most frequently is headache.

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.

**Question**: Find the day in which the symptom that occurs most frequently is headache.

** Query Plan**:

**  Preparation Steps : **
1. Initialize the process: Set up the environment and prepare for query execution, including initializing variables and temporary storage.
2. Open the symptom_search_sub_region_2_daily table: Access the table containing daily symptom search data.

** Extract the headache symptom: **
1. Scan the table: Start reading rows from the symptom_search_sub_region_2_daily table.
2. Identify the headache symptom: Look for the column containing the headache symptom data.
3. Extract the headache symptom value: For each row, extract the value from the headache symptom column.
4. Aggregate by date: Group the data by date to count the occurrences of the headache symptom on each day.

** Sort by frequency: **
1. Order the results in descending order of symptom frequency.
2. Limit the results: Extract the single day with the highest count.

** Step 2: Identify Subtasks **
1. Extract relevant symptom column: While "headache" is not explicitly listed, its frequency might be tracked in a related table (e.g., symptom_search_country_daily) as per the given gold query.
2. Group data by day of the week: Use FORMAT_DATE('%A', date) to extract the day of the week from each date.
3. Aggregate by count: Count the occurrences of the "headache" symptom across dates and group by the day of the week.
4. Sort by frequency: Order the results in descending order of symptom frequency.
5. Limit the results: Extract the single day with the highest count.

** Step 3: Formulate the Query **
1. From the subtasks, the query will:
2. Select the day of the week using FORMAT_DATE('%A', date).
3. Aggregate counts grouped by the day.
4. Sort the results by the aggregated count in descending order.
5. Limit the results to the top record.

** Step 4: Construct the Query **
1. Combining all subtasks, the final query is:
2. SELECT COUNT(symptom_headache) FROM {BQ_PROJECT_ID}.covid19_symptom_search.symptom_search_sub_region_2_daily GROUP BY FORMAT_DATE('%A', date) ORDER BY COUNT(symptom_headache) DESC LIMIT 1;

** Step 5: Finalize the Query **
**Final Optimized SQL Query:**
```sql
SELECT 
  FORMAT_DATE('%A', PARSE_DATE('%Y-%m-%d', date)) AS day,
  COUNT(*) AS headache_count
FROM 
  {BQ_PROJECT_ID}.covid19_symptom_search.symptom_search_country_daily
GROUP BY 
  day
ORDER BY 
  headache_count DESC
LIMIT 1;
```

Now is the real question, following the instruction and examples, generate the GoogleSQL with Recursive Divide-and-Conquer approach.
Follow all steps from the strategy. When you get to the final query, output the query string ONLY in the format ```sql ... ```. Make sure you only output one single query.

**************************
【Table creation statements】
{SCHEMA}

**************************
【Question】
Question:
{QUESTION}

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.
"""
