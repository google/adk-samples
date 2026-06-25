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

"""Divide-and-Conquer prompt template."""

DC_PROMPT_TEMPLATE = """
You are an experienced database expert.
Now you need to generate a GoogleSQL or BigQuery query given the database information, a question and some additional information.
The database structure is defined by table schemas (some columns provide additional column descriptions in the options).

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Here is a high level description of the steps.
1. **Divide (Decompose Sub-question with Pseudo SQL):** The complex natural language question is recursively broken down into simpler sub-questions. Each sub-question targets a specific piece of information or logic required for the final SQL query.
2. **Conquer (Real SQL for sub-questions):**  For each sub-question (and the main question initially), a "pseudo-SQL" fragment is formulated. This pseudo-SQL represents the intended SQL logic but might have placeholders for answers to the decomposed sub-questions.
3. **Combine (Reassemble):** Once all sub-questions are resolved and their corresponding SQL fragments are generated, the process reverses. The SQL fragments are recursively combined by replacing the placeholders in the pseudo-SQL with the actual generated SQL from the lower levels.
4. **Final Output:** This bottom-up assembly culminates in the complete and correct SQL query that answers the original complex question.

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
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.restaurant.generalinfo
(
 id_restaurant INT64,
 food_type STRING OPTIONS(description="the food type"),
 city STRING OPTIONS(description="the city where the restaurant is located in"),
);

CREATE TABLE `{BQ_DATA_PROJECT_ID}`.restaurant.location
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

**1. Divide and Conquer:**

* **Main Question:** How many Thai restaurants can be found in San Pablo Ave, Albany?
   * **Analysis:** The question asks for a count of restaurants, so we'll use `COUNT()` for that. The count should include only Thai restaurants, which we can identify using the `food_type` column in the `restaurant.generalinfo` table.  The location "San Pablo Ave, Albany" spans two columns (`street_name` and `city`) in the `restaurant.location` table, requiring us to join these two tables.
   * **Pseudo SQL:** SELECT COUNT(`T1`.`id_restaurant`) FROM `restaurantgeneralinfo` AS `T1` INNER JOIN `restaurant.location` AS `T2` ON `T1`.`id_restaurant` = `T2`.`id_restaurant` WHERE  <Thai restaurant> AND <in San Pablo Ave, Albany>

   * **Sub-question 1:** Thai restaurant
       * **Analysis:** This is a straightforward filter on the `restaurant.generalinfo` table using the `food_type` column.
       * **Pseudo SQL:** `T1`.`food_type` = 'thai'

   * **Sub-question 2:** in San Pablo Ave, Albany
       * **Analysis:** This location information is spread across two columns in the `restaurant.location` table. We need to combine these conditions with an "AND" operator to ensure both are met.
       * **Pseudo SQL:** `T2`.`street_name` = 'san pablo ave' AND `T2`.`city` = 'albany'

**2. Assembling SQL:**

* **Sub-question 1 (Thai restaurant):**
   * **SQL:** `T1`.`food_type` = 'thai'

* **Sub-question 2 (in San Pablo Ave, Albany):**
   * **SQL:** `T2`.`street_name` = 'san pablo ave' AND `T2`.`city` = 'albany'

* **Main Question (count of restaurants):**
   * **SQL:** SELECT COUNT(`T1`.`id_restaurant`) FROM `{BQ_DATA_PROJECT_ID}.restaurant.generalinfo` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.restaurant.location` AS `T2` ON `T1`.`id_restaurant` = `T2`.`id_restaurant` WHERE `T1`.`food_type` = 'thai' AND `T2`.`street_name` = 'san pablo ave' AND `T2`.`city` = 'albany'

**3. Simplification and Optimization:**

* The SQL query from step 2 is already quite efficient. We've used `INNER JOIN` to combine the tables based on their relationship, and the `WHERE` clause clearly defines our filtering criteria. There's no need for nested queries or complex sub-selections in this case.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(T1.id_restaurant)
 FROM `{BQ_DATA_PROJECT_ID}`.restaurant.generalinfo AS T1
 INNER JOIN `{BQ_DATA_PROJECT_ID}`.restaurant.location AS T2 ON T1.id_restaurant = T2.id_restaurant
 WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street_name = 'san pablo ave'
```

===========
Example 2

**************************
【Database Info】
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.financial.account (
   account_id INT64,
   district_id INT64,
   frequency STRING,
   date DATE,
);
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.financial.client (
   client_id INT64,
   gender STRING,
   birth_date DATE,
   district_id INT64,
);
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.financial.district (
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

**1. Divide and Conquer:**

* **Main Question:** What is the gender of the youngest client who opened account in the lowest average salary branch?
   * **Analysis:** The question is asking about `gender`, and it appears in the table `financial.client`. We will use this as the output column, selecting it from the youngest client in the lowest average salary branch.
   * **Pseudo **Final Optimized SQL Query:**** SELECT `T1`.`gender` FROM `{BQ_DATA_PROJECT_ID}.financial.client` AS `T1` WHERE <youngest client in the lowest average salary branch>

   * **Sub-question 1:** youngest client in the lowest average salary branch
       * **Analysis:** According to the hint, we need to use the `A11` from `financial.district` to get the salary info, and the youngest client can be obtained from using the `birth_date` column of table `financial.client`. The items between these two tables can be INNER JOIN using district_id.
       * **Pseudo SQL:** SELECT `T1`.`client_id` FROM `{BQ_DATA_PROJECT_ID}.financial.client` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.financial.district` AS `T2` ON `T1`.`district_id` = `T2`.`district_id` WHERE <lowest average salary branch> ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1

       * **Sub-question 1.1:** lowest average salary branch
           * **Analysis:** We can get the lowest average salary branch using order by `A11` ASC and pick top 1. The column `A11` is not NULLABLE, so we do not need to add "IS NOT NULL" filter
           * **Pseudo SQL:**  SELECT `district_id` FROM `{BQ_DATA_PROJECT_ID}.financial.district` ORDER BY `A11` ASC LIMIT 1

**2. Assembling SQL:**

* **Sub-question 1.1 (lowest average salary branch):**
   * **SQL:** SELECT `district_id` FROM `{BQ_DATA_PROJECT_ID}.financial.district` ORDER BY `A11` ASC LIMIT 1

* **Sub-question 1 (youngest client in the lowest average salary branch):**
   * **SQL:** SELECT `T1`.`client_id` FROM `{BQ_DATA_PROJECT_ID}.financial.client` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.financial.district` AS `T2` ON `T1`.`district_id` = `T2`.`district_id` WHERE `T2`.`district_id` IN (SELECT `district_id` FROM `financial.district` ORDER BY `A11` ASC LIMIT 1) ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1

* **Main Question (gender of the client):**
   * **SQL:** SELECT `T1`.`gender` FROM `{BQ_DATA_PROJECT_ID}.financial.client` AS `T1` WHERE `T1`.`client_id` = (SELECT `T1`.`client_id` FROM `{BQ_DATA_PROJECT_ID}.financial.client` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.financial.district` AS `T2` ON `T1`.`district_id` = `T2`.`district_id` WHERE `T2`.`district_id` IN (SELECT `district_id` FROM `{BQ_DATA_PROJECT_ID}.financial.district` ORDER BY `A11` ASC LIMIT 1) ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1)

**3. Simplification and Optimization:**

* The final SQL query from step 2 can be simplified and optimized. The nested queries can be combined using a single `INNER JOIN` and the filtering can be done within a single `ORDER BY` clause.

**Final Optimized SQL Query:**
```sql
SELECT `T1`.`gender`
 FROM `{BQ_DATA_PROJECT_ID}.financial.client` AS `T1`
 INNER JOIN `{BQ_DATA_PROJECT_ID}.financial.district` AS `T2`
 ON `T1`.`district_id` = `T2`.`district_id`
 ORDER BY `T2`.`A11` ASC, `T1`.`birth_date` DESC NULLS LAST
 LIMIT 1
```
===========
Example 3 (dividing into two parallel sub-questions)

**************************
【Database Info】
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.olympics.games
(
 id INT64,
 games_year INT64 OPTIONS(description="description: the year of the game"),
 games_name STRING,
);

CREATE TABLE `{BQ_DATA_PROJECT_ID}`.olympics.games_city
(
 games_id INT64,
 city_id INT64 OPTIONS(description="the id of the city that held the game Maps to city(id)"),
);

CREATE TABLE `{BQ_DATA_PROJECT_ID}`.olympics.city
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

**1. Divide and Conquer:**

* **Main Question:** From 1900 to 1992, how many games did London host?
   * **Analysis:** The question requires us to count games, which are represented by the `id` column in the `olympics.games` table.  We need to filter these games based on two criteria: they were hosted in London and occurred between 1900 and 1992.
   * **Pseudo SQL:** SELECT COUNT(`T1`.`id`) FROM `{BQ_DATA_PROJECT_ID}.olympics.games` AS `T1`  WHERE  <games are in London> AND <games year between 1900 and 1992>

   * **Sub-question 1:** games are in London
       * **Analysis:**  To determine which games were hosted in London, we need to join the `olympics.games` table with the `olympics.games_city` table on `games_id` and then join with the `city` table on `city_id`. We'll use `INNER JOIN` to ensure only matching records are considered.  The filtering on 'London' will be applied to the `city_name` column.
       * **Pseudo SQL:**  `T1`.`id` IN (SELECT `T1`.`games_id` FROM `{BQ_DATA_PROJECT_ID}.olympics.games_city` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.olympics.city` AS `T2` ON `T1`.`city_id` = `T2`.`id` WHERE `T2`.`city_name` = 'London')

   * **Sub-question 2:** games year between 1900 and 1992
       * **Analysis:** This involves filtering the `olympics.games` table directly based on the `games_year` column using the `BETWEEN` operator.
       * **Pseudo SQL:** `T1`.`games_year` BETWEEN 1900 AND 1992

**2. Assembling SQL:**

* **Sub-question 1 (games are in London):**
   * **SQL:**  `T1`.`id` IN (SELECT `T1`.`games_id` FROM `{BQ_DATA_PROJECT_ID}.olympics.games_city` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.olympics.city` AS `T2` ON `T1`.`city_id` = `T2`.`id` WHERE `T2`.`city_name` = 'London')

* **Sub-question 2 (games year between 1900 and 1992):**
   * **SQL:**  `T1`.`games_year` BETWEEN 1900 AND 1992

* **Main Question (count of games):**
   * **SQL:** SELECT COUNT(`T1`.`id`) FROM `{BQ_DATA_PROJECT_ID}.olympics.games` AS `T1` WHERE `T1`.`id` IN (SELECT `T1`.`games_id` FROM `{BQ_DATA_PROJECT_ID}.olympics.games_city` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.olympics.city` AS `T2` ON `T1`.`city_id` = `T2`.`id` WHERE `T2`.`city_name` = 'London') AND `T1`.`games_year` BETWEEN 1900 AND 1992

**3. Simplification and Optimization:**

* The nested query can be converted into a more efficient `JOIN` operation. We'll use `INNER JOIN` to combine ``{BQ_DATA_PROJECT_ID}`.olympics.games`, ``{BQ_DATA_PROJECT_ID}`.olympics.games_city`, and ``{BQ_DATA_PROJECT_ID}`.olympics.city` based on the relationships between them.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(T3.id)
 FROM `{BQ_DATA_PROJECT_ID}`.olympics.games_city AS T1
 INNER JOIN `{BQ_DATA_PROJECT_ID}`.olympics.city AS T2 ON T1.city_id = T2.id
 INNER JOIN `{BQ_DATA_PROJECT_ID}`.olympics.games AS T3 ON T1.games_id = T3.id
 WHERE T2.city_name = 'London' AND T3.games_year
 BETWEEN 1900 AND 1992
```

===========
Example 4 (When it's not clear which column should be used for a string matching, use a loosen condition such as string LIKE and OR condition to cover multiple possible columns.)

**************************
【Database Info】
CREATE TABLE `{BQ_DATA_PROJECT_ID}.academics.student_programs` (
   `Program Type` STRING,
   `Participants (Ages 10-15)` FLOAT64,
   `Total Enrollment (Ages 10-15)` FLOAT64,
   `School Category` STRING,
);

**************************
【Question】
Question:
Please list the lowest three participation rates for students aged 10-15 in online programs. Participation rate for students aged 10-15 = `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`

**************************
【Answer】
Repeating the question and generating the SQL with Recursive Divide-and-Conquer.
**Question:** Please list the lowest three participation rates for students aged 10-15 in online programs. Participation rate for students aged 10-15 = `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`

**1. Divide and Conquer:**

* **Main Question:** Please list the lowest three participation rates for students aged 10-15 in online programs.
   * **Analysis:** The question is asking about the ratio between `Participants (Ages 10-15)` and `Total Enrollment (Ages 10-15)`. We need to filter the data to only include online programs.
   * **Pseudo SQL:** SELECT (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) FROM `{BQ_DATA_PROJECT_ID}.academics.student_programs` WHERE <online programs> ORDER BY (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) ASC NULLS LAST LIMIT 3

   * **Sub-question 1:** online programs
       * **Analysis:** We will get the information from the table `{BQ_DATA_PROJECT_ID}.academics.student_programs`.
       * **Pseudo SQL:** SELECT program_id FROM `academics.student_programs` WHERE <condition for online programs>

       * **Sub-question 1.1:** condition for online programs (Note: This requires external knowledge or database schema information. We need to identify which column(s) indicate "online programs".)
           * **Analysis:** We'll assume either "School Category" or "Program Type" columns might contain the term "online."
           * **Pseudo SQL:**  LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'

**2. Assembling SQL:**

* **Sub-question 1.1 (condition for online programs):**
   * **SQL:** LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'

* **Sub-question 1 (online programs):**
   * **SQL:** SELECT program_id FROM `{BQ_DATA_PROJECT_ID}.academics.student_programs` WHERE LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'

* **Main Question (lowest three participation rates):**
   * **SQL:** SELECT (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) FROM `{BQ_DATA_PROJECT_ID}.academics.student_programs` WHERE program_id IN (SELECT program_id FROM `{BQ_DATA_PROJECT_ID}.academics.student_programs` WHERE LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%') ORDER BY (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) ASC NULLS LAST LIMIT 3

**3. Simplification and Optimization:**

* We can directly incorporate the condition for online programs into the main query.

**Final Optimized SQL Query:**
```sql
SELECT `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)` FROM `{BQ_DATA_PROJECT_ID}.academics.student_programs`
 WHERE LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'
 AND `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)` IS NOT NULL
 ORDER BY `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)` ASC NULLS LAST LIMIT 3;
```

===========
Example 5

**************************
【Database Info】
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.retails.employees (
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

**1. Divide and Conquer:**

* **Main Question:** How many employees earn over $100,000?

   * **Pseudo SQL:** SELECT COUNT(*) FROM {BQ_DATA_PROJECT_ID}.retails.employees WHERE <employees earning over 100000>
   * **Analysis:** The question is asking about the COUNT of employees. We need to filter the data to only include employees earning over $100,000.

   * **Sub-question 1:** employees earning over 100000
       * **Analysis:** Simple condition on the `salary` column.
       * **Pseudo SQL:** SELECT employee_id FROM {BQ_DATA_PROJECT_ID}.retails.employees WHERE salary > 100000

**2. Assembling SQL:**

* **Sub-question 1 (employees earning over 100000):**
   * **SQL:** SELECT employee_id FROM `{BQ_DATA_PROJECT_ID}`.retails.employees WHERE salary > 100000

* **Main Question (count of employees):**
   * **SQL:** SELECT COUNT(*) FROM `{BQ_DATA_PROJECT_ID}`.retails.employees WHERE employee_id IN (SELECT employee_id FROM `{BQ_DATA_PROJECT_ID}`.retails.employees WHERE salary > 100000)

**3. Simplification and Optimization:**

* We can achieve the same result more efficiently within a single WHERE clause.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(*) FROM `{BQ_DATA_PROJECT_ID}`.retails.employees WHERE salary > 100000;
```

===========
Example 6

**************************
【Database Info】
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.airlines.Airlines
(
 FL_DATE STRING OPTIONS(description="flight date"),
 ORIGIN STRING OPTIONS(description="airport of origin"),
 DEST STRING OPTIONS(description="Destination airport"),
);

CREATE TABLE `{BQ_DATA_PROJECT_ID}`.airlines.Airports
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

**1. Divide and Conquer:**

* **Main Question:** How many flights were there from San Diego International airport to Los Angeles International airport in the August of 2018?
   * **Analysis:** The question asks for a count of flights, which can be obtained by counting the `FL_DATE` entries in the `airlines.Airlines` table. We need to apply three filters: flights originating from San Diego International, flights destined for Los Angeles International, and flights occurring in August 2018.
   * **Pseudo SQL:** SELECT COUNT(`FL_DATE`) FROM `{BQ_DATA_PROJECT_ID}.airlines.Airlines` WHERE <flights are in August 2018> AND <flights are from San Diego International> AND <flights are to Los Angeles International>

   * **Sub-question 1:** flights are in August 2018
       * **Analysis:** This filter can be directly applied to the `{BQ_DATA_PROJECT_ID}.airlines.Airlines` table using the `FL_DATE` column and the `LIKE` operator, as indicated by the evidence.
       * **Pseudo SQL:** `FL_DATE` LIKE '2018/8%'

   * **Sub-question 2:** flights are from San Diego International
       * **Analysis:**  We need to find the airport code (`ORIGIN`) corresponding to 'San Diego, CA: San Diego International' from the `{BQ_DATA_PROJECT_ID}.airlines.Airports` table and use it to filter the `airlines.Airlines` table. This requires joining `airlines.Airports` and `airlines.Airlines` based on `airlines.Airports`.`Code` = `airlines.Airlines`.`ORIGIN`.
       * **Pseudo SQL:** `ORIGIN` = (SELECT `T2`.`ORIGIN` FROM `{BQ_DATA_PROJECT_ID}.airlines.Airports` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.airlines.Airlines` AS `T2` ON `T1`.`Code` = `T2`.`ORIGIN` WHERE `T1`.`Description` = 'San Diego, CA: San Diego International')

   * **Sub-question 3:** flights are to Los Angeles International
       * **Analysis:** Similar to sub-question 2, we need to find the airport code (`DEST`) for 'Los Angeles, CA: Los Angeles International' from the `airlines.Airports` table and use it to filter the `airlines.Airlines` table. This also requires joining `airlines.Airports` and `airlines.Airlines`, but this time on `airlines.Airports`.`Code` = `airlines.Airlines`.`DEST`.
       * **Pseudo SQL:** `DEST` = (SELECT `T4`.`DEST` FROM `{BQ_DATA_PROJECT_ID}.airlines.Airports` AS `T3` INNER JOIN `{BQ_DATA_PROJECT_ID}.airlines.Airlines` AS `T4` ON `T3`.`Code` = `T4`.`DEST` WHERE `T3`.`Description` = 'Los Angeles, CA: Los Angeles International')

**2. Assembling SQL:**

* **Sub-question 1 (flights are in August 2018):**
   * **SQL:** `FL_DATE` LIKE '2018/8%'

* **Sub-question 2 (flights are from San Diego International):**
   * **SQL:** `ORIGIN` = (SELECT DISTINCT `T2`.`ORIGIN` FROM `{BQ_DATA_PROJECT_ID}.airlines.Airports` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.airlines.Airlines` AS `T2` ON `T1`.`Code` = `T2`.`ORIGIN` WHERE `T1`.`Description` = 'San Diego, CA: San Diego International')

* **Sub-question 3 (flights are to Los Angeles International):**
   * **SQL:** `DEST` = (SELECT DISTINCT `T4`.`DEST` FROM `{BQ_DATA_PROJECT_ID}.airlines.Airports` AS `T3` INNER JOIN `{BQ_DATA_PROJECT_ID}.airlines.Airlines` AS `T4` ON `T3`.`Code` = `T4`.`DEST` WHERE `T3`.`Description` = 'Los Angeles, CA: Los Angeles International')

* **Main Question (count of flights):**
   * **SQL:** SELECT COUNT(`FL_DATE`) FROM `{BQ_DATA_PROJECT_ID}.airlines.Airlines` WHERE `FL_DATE` LIKE '2018/8%' AND `ORIGIN` = (SELECT `T2`.`ORIGIN` FROM `{BQ_DATA_PROJECT_ID}.airlines.Airports` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.airlines.Airlines` AS `T2` ON `T1`.`Code` = `T2`.`ORIGIN` WHERE `T1`.`Description` = 'San Diego, CA: San Diego International') AND `DEST` = (SELECT `T4`.`DEST` FROM `{BQ_DATA_PROJECT_ID}.airlines.Airports` AS `T3` INNER JOIN `{BQ_DATA_PROJECT_ID}.airlines.Airlines` AS `T4` ON `T3`.`Code` = `T4`.`DEST` WHERE `T3`.`Description` = 'Los Angeles, CA: Los Angeles International')

**3. Simplification and Optimization:**

* The query in step 2 is already quite optimized. We are using nested queries to avoid joining the `airlines.Airports` table multiple times in the main query, which could potentially impact performance.

**Final Optimized SQL Query:**
```sql
SELECT COUNT(FL_DATE)
 FROM `{BQ_DATA_PROJECT_ID}`.airlines.Airlines
 WHERE FL_DATE LIKE '2018/8%'
 AND ORIGIN = (
   SELECT DISTINCT T2.ORIGIN
   FROM `{BQ_DATA_PROJECT_ID}`.airlines.Airports AS T1
   INNER JOIN `{BQ_DATA_PROJECT_ID}`.airlines.Airlines AS T2 ON T1.Code = T2.ORIGIN
   WHERE T1.Description = 'San Diego, CA: San Diego International' )
 AND DEST = (
   SELECT DISTINCT T4.DEST
   FROM `{BQ_DATA_PROJECT_ID}`.airlines.Airports AS T3
   INNER JOIN `{BQ_DATA_PROJECT_ID}`.airlines.Airlines AS T4 ON T3.Code = T4.DEST
   WHERE T3.Description = 'Los Angeles, CA: Los Angeles International' )
```

===========
Example 7

**************************
【Database Info】
CREATE TABLE `{BQ_DATA_PROJECT_ID}`.food_inspection.businesses
(
       `business_id` INT64,
       `name` STRING OPTIONS(description="the name of the eatery"),
);

CREATE TABLE `{BQ_DATA_PROJECT_ID}`.food_inspection.inspections
(
       `business_id` INT64 OPTIONS(description="the unique id of the business"),
       `score` INT64 OPTIONS(description="description: the inspection score"),
       `date` DATE,
);

CREATE TABLE `{BQ_DATA_PROJECT_ID}`.food_inspection.violations
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

**1. Divide and Conquer:**

* **Main Question:** What are the names of the establishments that met all the required standards for 4 consecutive years?
   * **Analysis:** We need to find the names of businesses that have a score of 100 for 4 consecutive years. The `food_inspection.businesses` table contains the `name` and the `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` table contains the `score` and `date`. We will need to join these tables and filter by score. To check for consecutive years, we'll need to group by business and year, then check if each group has a count of 4.
   * **Pseudo SQL:** SELECT DISTINCT `T2`.`name` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE  <score = 100> AND <4 consecutive years>

   * **Sub-question 1:** score = 100
       * **Analysis:** This is a simple filter on the `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` table where we select rows with a `score` of 100.
       * **Pseudo SQL:** `T1`.`score` = 100

   * **Sub-question 2:** 4 consecutive years
       * **Analysis:** This is more complex. We need to group the inspections by business and year, then check if the count for each group is 4. To get the year from the `date` column, we'll use the `FORMAT_DATE('%Y', date)` function. We'll also need to use window functions to assign a rank to each year within a business, allowing us to check for consecutiveness.
       * **Pseudo SQL:** `T2`.`name` IN (SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4)

       * **Sub-question 2.1:** Get distinct businesses and their inspection years where the score is 100
           * **Analysis:** We need to join `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` and `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` tables, filter by `score` = 100, and select distinct business names and their inspection years.
           * **Pseudo SQL:** SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100

       * **Sub-question 2.2:** Assign a rank to each year within a business
           * **Analysis:** We'll use the `row_number()` window function to assign a rank to each year within each business, ordered chronologically. This will help us identify consecutive years later.
           * **Pseudo SQL:** SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100` AS `T3`

       * **Sub-question 2.3:** Group by business and consecutive year groups and check if the count is 4
           * **Analysis:** We'll group the results by business name and a calculated date representing the start of each potential 4-year period. This date is calculated by adding (`rowNumber` - 1) years to the first day of the year extracted from the `years` column. We then filter for groups with a count of 4, indicating 4 consecutive years.
           * **Pseudo SQL:** SELECT `T4`.`name` FROM (<previous sub-query>) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4

**2. Assembling SQL:**

* **Sub-question 2.1 (distinct businesses and years with score 100):**
   * **SQL:** SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100

* **Sub-question 2.2 (assign rank to each year within a business):**
   * **SQL:** SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`

* **Sub-question 2.3 (group by business and consecutive year groups):**
   * **SQL:** SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, DATE_SUB(DATE(CONCAT(T4.years, '-01-01')), INTERVAL (T4.rowNumber - 1) YEAR)  HAVING COUNT(`T4`.`years`) = 4

* **Sub-question 2 (4 consecutive years):**
   * **SQL:** `T2`.`name` IN (SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, DATE_SUB(DATE(CONCAT(T4.years, '-01-01')), INTERVAL (T4.rowNumber - 1) YEAR)  HAVING COUNT(`T4`.`years`) = 4)

* **Main Question (names of establishments):**
   * **SQL:** SELECT DISTINCT `T2`.`name` FROM `{BQ_DATA_PROJECT_ID}.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE  `T1`.`score` = 100 AND `T2`.`name` IN (SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, FORMAT_DATE('%Y', date) AS `years` FROM `{BQ_DATA_PROJECT_ID}.food_inspection.inspections` AS `T1` INNER JOIN `{BQ_DATA_PROJECT_ID}.food_inspection.businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, DATE_SUB(DATE(CONCAT(T4.years, '-01-01')), INTERVAL (T4.rowNumber - 1) YEAR) HAVING COUNT(`T4`.`years`) = 4)

**3. Simplification and Optimization:**

* The final SQL query from step 2 can be simplified by merging the nested queries into a single query with a `WITH` clause. This improves readability and potentially performance.

**Final Optimized SQL Query:**
```sql
SELECT DISTINCT T4.name
 FROM ( SELECT T3.name, T3.years, row_number()
 OVER (PARTITION BY T3.name ORDER BY T3.years)
 AS rowNumber FROM ( SELECT DISTINCT name, FORMAT_DATE('%Y', date)
 AS years FROM `{BQ_DATA_PROJECT_ID}`.food_inspection.inspections AS T1
 INNER JOIN `{BQ_DATA_PROJECT_ID}`.food_inspection.businesses AS T2 ON T1.business_id = T2.business_id
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

Analysis: We need to determine the day (day of the week) when the frequency of searches for the symptom "headache" is the highest. This involves:
   - Grouping the data by the day of the week.
   - Counting the occurrences of searches for "headache."
   - Sorting the counts in descending order and selecting the day with the highest count.

Pseudo SQL:
   SELECT FORMAT_DATE('%A', date) AS day, COUNT(*) AS headache_count
   FROM `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`
   WHERE symptom_Headache > 0
   GROUP BY day
   ORDER BY headache_count DESC
   LIMIT 1

Sub-question 1: Extract the day of the week from the date column.
   - Analysis: Use the FORMAT_DATE function with the %A format specifier to extract the day name (e.g., "Monday," "Tuesday") from the date column.

Pseudo SQL:
   SELECT FORMAT_DATE('%A', date) AS day
   FROM `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`

Sub-question 2: Filter rows where "headache" searches occurred.
   - Analysis: Only include rows where the symptom "headache" has a positive value (symptom_Headache > 0).

Pseudo SQL:
   SELECT date
   FROM `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`
   WHERE symptom_Headache > 0

Sub-question 3: Count the occurrences of "headache" searches grouped by day of the week.
   - Analysis: After filtering the data for rows where symptom_Headache > 0, group the data by the day of the week and count the number of rows for each day.

Pseudo SQL:
   SELECT FORMAT_DATE('%A', date) AS day, COUNT(*) AS headache_count
   FROM `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`
   WHERE symptom_Headache > 0
   GROUP BY day

Sub-question 4: Sort the results by the count in descending order and get the top day.
   - Analysis: Use the ORDER BY clause to sort by the count of "headache" searches in descending order. Limit the result to 1 to get the top day.

Pseudo SQL:
   SELECT FORMAT_DATE('%A', date) AS day, COUNT(*) AS headache_count
   FROM `bigquery-public-data.covid19_symptom_search.symptom_search_sub_region_2_daily`
   WHERE symptom_Headache > 0
   GROUP BY day
   ORDER BY headache_count DESC
   LIMIT 1

Assembling SQL
   - Combining all sub-questions into the final query:

**Final Optimized SQL Query:**
```sql
SELECT
  FORMAT_DATE('%A', PARSE_DATE('%Y-%m-%d', date)) AS day,
  COUNT(*) AS headache_count
FROM
  `bigquery-public-data`.`covid19_symptom_search`.`symptom_search_country_daily`
GROUP BY
  day
ORDER BY
  headache_count DESC
LIMIT 1;
```

Now is the real question, following the instruction and examples, generate the GoogleSQL with Recursive Divide-and-Conquer approach.
Follow all steps from the strategy. When you get to the final query, output the query string ONLY in the format ```sql ... ```. Make sure you only output one single query.
Table names always should be exactly the same as the table names mentioned in the database schema, for example, `{BQ_DATA_PROJECT_ID}.airlines.Airlines` instead of `Airlines`.

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
