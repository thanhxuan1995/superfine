prompt_country = """given the video game advertisement input data as below, do not give reasoning contents,   
            directly provide the answer as requested without the code explanation.

            Follow this JSON format strictly:
            {format_instruction}

            Your tasks:
            1. Extract `campaign_id`, `{your_column_input}`, and revenue.
            2. **If `{your_column_input}` does not exist, print only:** "we can not find the matching column name" **and stop.**
            3. Use an **80% confidence interval** for `{your_column_input}`.
            4. Identify the **top half (50%)** of `{your_column_input}` values that have the highest confidence in generating high revenue.
            5. **Ensure the number of values selected is 50% of the unique `{your_column_input}` values present in the dataset.**
            6. **Ensure the final list contains **only unique country codes** in **ISO Alpha-2 format** (e.g., "US", "VN", "FR")**
            7. Directly output only the final list in JSON format as follows:

            Expected Output Format:
            {{"result": {{"{your_column_input}": ["US", "VN", "FR", "JP", "DE", "IT", ...]}}}}

            **Note:**  
            - The selection must be **dynamically based on the dataset** and not a static list.  
            - If the dataset has **an odd number of values**, round **up** to include more data.
            - **Do NOT use full country names**, only return **two-letter ISO country codes**.

            Input Data: {data}

            Output:
            """


prompt_bidvalue = """given the video game advertisement input data as below, do not give reasoning contents,  
            directly provide the answer as requested without the code explanation.

            Follow this JSON format strictly:
            {format_instruction}

            Your tasks:
            1. Extract `campaign_id`, `{main_col}`, and revenue.
            2. **If `{main_col}` does not exist, print only:** "we can not find the matching column name" **and stop.**
            3. Identify `{sub_colmn}` values **exactly as they appear** in `{main_col}`, ensuring they remain as **decimal values** (e.g., `0.023`, `0.093`).
            4. Retrieve the corresponding revenues for each `{sub_colmn}` value.
            5. Apply an **80% confidence interval** to select **the top half** of `{sub_colmn}` values that have the highest confidence in maximizing revenue.
            6. Ensure all `{sub_colmn}` values remain **unchanged** (no scaling, rounding, or formatting modifications).
            7. **Do NOT include invalid numbers or unrelated values. Ensure selection contains **only unique** `{sub_colmn}` values.**
            8. Output only the final JSON result in the format below:

            Expected Output Format:
            {{"result": {{"{sub_colmn}": ["0.023", "0.093", "0.042", "0.015", "0.078", ...]}}}}  

            **Note:**  
            - `{sub_colmn}` values **must be taken exactly as they appear in `{main_col}`**.  
            - **Do not round or scale the values** beyond their original dataset format.  
            - Ensure the output list contains **half of the bid values** from the dataset.  

            Input Data: {data}

            Output:
            """
prompt_gender = """given the video game advertisement input data as below, do not give reasoning contents,  
            directly provide the answer as requested without the code explanation.

            Follow this JSON format strictly:
            {format_instruction}

            Your tasks:
            1. Extract `campaign_id`, `{yr_col}`, and revenue.
            2. **If `{yr_col}` does not exist, print only:** "all gender" **and stop.**
            3. Identify `{sub_item}` values **exactly as they appear** in `{yr_col}`, ensuring they remain as values as they are.
            4. Retrieve the corresponding revenues for each `{sub_item}` value.
            5. Apply an **80% confidence interval** to select **the top half** of `{sub_item}` values that have the highest confidence in maximizing revenue.
            6. Ensure all `{sub_item}` values remain **unchanged** (no scaling, rounding, or formatting modifications).
            7. **Do NOT include unrelated values.**. Ensure selection contains **only unique** `{sub_item}` values.
            8. Output only the final JSON result in the format below:

            Expected Output Format:
            {{"result": {{"{sub_item}": ["girl", "boy", "adult", ...]}}}}  

            **Note:**  
            - `{sub_item}` values **must be taken exactly as they appear in `{yr_col}`**.    
            - Ensure the output list contains **half of the values** from the `{sub_item}` values in the dataset.  

            Input Data: {data}

            Output:
            """

