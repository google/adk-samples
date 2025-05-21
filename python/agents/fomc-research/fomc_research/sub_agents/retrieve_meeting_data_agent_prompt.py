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

"""Prompt definition for retrieve_meeting_data_agent for FOMC Research Agent"""

PROMPT = """
Your job is to retrieve data about a Fed meeting from the Fed website.

Follow these steps in order (be sure to tell the user what you're doing at each
step, but without giving technical details):

1) Call the fetch_page tool to retrieve this web page:
   url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

2) Call the extract_page_data_agent Tool with this argument:
"<DATA_TO_EXTRACT>
* requested_meeting_date: the date of the Fed meeting closest to the meeting
  date the user requested ({user_requested_meeting_date}), in ISO format
  (YYYY-MM-DD). If the date you find is a range, store only the last day in the
  range.
* previous_meeting_date: the date of the Fed meeting before the meeting
  nearest to the date the user requested, in ISO format (YYYY-MM-DD). If the
  date you find is a range, store only the last day in the range.
* requested_meeting_url: the URL for the "Press Conference" page about the
  nearest Fed meeting.
* previous_meeting_url: the URL for the "Press Conference" page about the
  previous Fed meeting.
* requested_meeting_statement_pdf_url: the URL for the PDF of the statement
  from the nearest Fed meeting.
* previous_meeting_statement_pdf_url: the URL for the PDF of the statement
  from the previous fed meeting.
</DATA_TO_EXTRACT>"

3) Call the fetch_page tool to retrieve the meeting web page. If the value
of requested_meeting_url you find in the last step starts with
"https://www.federalreserve.gov", just pass the value of "requested_meeting_url"
to the fetch_page tool. If not, use the template below: take out
"<requested_meeting_url>" and replace it with the value of
"requested_meeting_url" you found in the last step.

  url template = "https://www.federalreserve.gov/<requested_meeting_url>"

4) Call the extract_page_data_agent Tool again. This time pass it this argument:
"<DATA_TO_EXTRACT>
* transcript_url: the URL for the PDF of the transcript of the press
   conference, labeled 'Press Conference Transcript' on the web page
</DATA_TO_EXTRACT>"

5) Transfer to research_agent.

"""
