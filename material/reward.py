import re
import json
import wandb
from loguru import logger

from llm.factory import LLMInterface

llm_client_bk = LLMInterface("openai", "o3-mini")
llm_client = LLMInterface("openai", "o3-mini")
# llm_client = LLMInterface("bedrock", "us.deepseek.r1-v1:0")


PROBLEM_ANALYSIS_EVALUATION_PROMPT = """You are evaluating the response's analysis quality of a knowledge graph.

## Knowledge Graph:
{graph}

## Reference Answer:
{reference_answer}

## The Response to Evaluate (Make sure not to mix up your answers with the reference answer):
{response}

## Evaluation Approach

Your task is to evaluate the quality of the response's analysis based on the reference answer and using your own judgment by carefully examining the actual knowledge graph data:

You need to traverse the analysis points (surrounded by <analysis> tags) in the response and evaluate each analysis point on a scale of 0-1:

* 1.0 points: Analysis correctly identifies a genuine issue with thorough, logical analysis that is fully supported by the knowledge graph data
* 0.6 points: Analysis correctly identifies a genuine issue with mostly accurate analysis but some gaps or minor inconsistencies with the data
* 0.2 points: Analysis correctly identifies a genuine issue but analysis is superficial or has significant issues
* 0.0 points: Analysis is factually incorrect (contradicts the knowledge graph data), or analysis is completely flawed, or the issue has already been analyzed before

IMPORTANT: You MUST verify every analysis point by comparing it with the actual knowledge graph data provided. Never accept the response's claims without verification, especially when:
1. The analysis point is not mentioned in the reference answer
2. The analysis point contradicts something in the reference answer

For factual verification:
- Check that entity/relationship IDs actually exist in the graph
- Verify that claimed redundancies or issues are genuine by examining the actual content
- Confirm that analyses of relationships accurately reflect the true source_entity_id, target_entity_id, and description

Note: Duplicate analysis points (analyzing the same issue multiple times) will receive 0 points. Each unique issue should only be analyzed once.

Use the reference answer as a guide, but also apply your judgment for potentially valid issues not in the reference. However, all such additional issues must be rigorously verified against the actual knowledge graph data.

Deduct -1 point for obvious point farming (raising multiple invalid issues).

## Score Calculation Method

For each analysis point, briefly justify your score.
1. Start with a base score point of 0.0
2. Add score points for each identified issue according to the evaluation approach above (0-1 score point scale for each analysis point)
3. Apply deductions for hack attempts if applicable, -1 score point
4. Sum up the scores of all analysis points to get the final total score

Example calculation:
- Response correctly identifies 3 unique issues with thorough analysis (+3.0 points)
- Response identifies 1 issue point with superficial analysis (+0.2 points)
- No hack attempts
- Total score = 3.0 + 0.2 = 3.2

## Output Format Instructions

Your evaluation should be structured as follows:

1. First, begin your detailed analysis with a <think> tag.
2. Within the <think> tags:
   - For each analysis point, evaluate the quality of the analysis provided, give your thought process and the score you give for each analysis point
3. End your analysis with </think> tag.
4. Finally, provide your final evaluation score in JSON format (surrounded by ```json and ``` markers).

The JSON must include this field:
- "total_analysis_quality_score": the total score of all analysis points in float format

Example of proper output format:

<think>

Response identifies 4 analysis points:
...
The reponse contains > 3 irrelevant and hacky analysis points, so I deduct 1 point.
Total score: 1.0 + 0.6 + 1.0 + 0.2 - 1 = 1.8
</think>

```json 
{{
  "total_analysis_quality_score": 1.8
}}
```

Ensure your evaluation maintains consistent standards across different responses.
"""


def compute_f1_score(student_ids_set, reference_ids_set):
    if len(student_ids_set) == 0 or len(reference_ids_set) == 0:
        return 0
    intersection_count = len(student_ids_set.intersection(reference_ids_set))
    precision_score = intersection_count / len(student_ids_set)
    recall_score = intersection_count / len(reference_ids_set)
    f1_score = 0
    if precision_score + recall_score > 0:
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
    return f1_score


def normalize_affected_ids(redundancy_sets):
    # Improved set merging algorithm for entity redundancy sets
    merged_sets = []
    for entity_set in redundancy_sets:
        # Find all existing sets that overlap with the current set
        overlapping_sets = []
        non_overlapping_sets = []

        for existing_set in merged_sets:
            if not entity_set.isdisjoint(existing_set):
                overlapping_sets.append(existing_set)
            else:
                non_overlapping_sets.append(existing_set)

        if overlapping_sets:
            # Create a new merged set containing all overlapping sets plus the current set
            new_merged_set = entity_set.copy()
            for overlap_set in overlapping_sets:
                new_merged_set.update(overlap_set)

            # Replace all overlapping sets with the new merged set
            merged_sets = non_overlapping_sets + [new_merged_set]
        else:
            # No overlaps, add as a new set
            merged_sets.append(entity_set.copy())

    return merged_sets


def extract_json(response: str) -> str:
    """Extract JSON from the plan response."""
    json_code_block_pattern = re.compile(
        r"```json\s*(\[\s*{.*?}\s*\])\s*```", re.DOTALL
    )
    match = json_code_block_pattern.search(response)
    if match:
        return match.group(1)

    json_code_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_code_block_pattern.search(response)
    if match:
        return match.group(1)

    return None


def extract_response(response: str) -> str:
    """Extract the response from the response."""
    sections = response.split("</think>")
    return sections[1].strip() if len(sections) > 1 else ""


def extract_thinking(response: str) -> str:
    """Extract the thinking from the response."""
    sections = response.split("</think>")
    return sections[0].strip() if len(sections) > 1 else ""


def extract_issues(response: str):
    analysis_tags = re.findall(
        r'<analysis\s*[">]?\s*(.*?)\s*</analysis>', response, re.DOTALL
    )

    entity_redundancy_issues = []
    relationship_redundancy_issues = []
    entity_quality_issues = []
    relationship_quality_issues = []

    # Process each analysis tag
    for analysis in analysis_tags:
        # Extract issue_type and affected_ids
        issue_type_match = re.search(r"issue_type:\s*([^\n]*)", analysis)
        affected_ids_match = re.search(r"affected_ids:\s*\[(.*?)\]", analysis)
        reasoning_match = re.search(r"reasoning:\s*([\s\S]*?)\n", analysis)
        conclusion_match = re.search(r"conclusion:\s*([\s\S]*?)\n", analysis)
        confidence_match = re.search(r"confidence:\s*([\w_/]+)\n", analysis)
        facto_search_match = re.search(r"facto_search:\s*([\s\S]*?)\n", analysis)

        if (
            not issue_type_match
            or not affected_ids_match
            or not reasoning_match
            or not conclusion_match
            or not confidence_match
        ):
            continue

        issue_type = issue_type_match.group(1).strip()
        affected_ids_str = affected_ids_match.group(1).strip()
        try:
            affected_ids = [int(id.strip()) for id in affected_ids_str.split(",")]
        except ValueError:
            continue

        reasoning = reasoning_match.group(1).strip()
        conclusion = conclusion_match.group(1).strip()
        confidence = confidence_match.group(1).strip()
        if facto_search_match:
            facto_search = facto_search_match.group(1).strip()
        else:
            facto_search = ""

        issue = {
            "issue_type": issue_type,
            "affected_ids": affected_ids,
            "reasoning": reasoning,
            "conclusion": conclusion,
            "confidence": confidence,
            "facto_search": facto_search,
        }
        # Categorize by issue type
        if issue_type == "redundancy_entity":
            entity_redundancy_issues.append(issue)
        elif issue_type == "redundancy_relationship":
            relationship_redundancy_issues.append(issue)
        elif issue_type == "entity_quality_issue":
            entity_quality_issues.append(issue)
        elif issue_type == "relationship_quality_issue":
            relationship_quality_issues.append(issue)

    return {
        "entity_redundancy_issues": entity_redundancy_issues,
        "relationship_redundancy_issues": relationship_redundancy_issues,
        "entity_quality_issues": entity_quality_issues,
        "relationship_quality_issues": relationship_quality_issues,
    }


def strict_format_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the response contains valid graph optimization actions."""
    responses = [completion[0]["content"] for completion in completions]
    scores = []
    print(kwargs)
    ids = kwargs["id"]

    for i, response in enumerate(responses):
        logger.info(f"Compute Strict Format Reward for {ids[i]}")
        this_score = 0
        try:
            this_score += (
                2
                - abs(response.count("<think>") - 1)
                - abs(response.count("</think>") - 1)
            ) * 0.2

            answer_sections = response.split("</think>")
            if len(answer_sections) != 2:
                scores.append(this_score)
                continue

            analysis_start_count = response.count("<analysis>")
            analysis_end_count = response.count("</analysis>")
            if analysis_start_count != analysis_end_count:
                this_score -= abs(analysis_start_count - analysis_end_count) * 0.1

            analysis_tags = re.findall(
                r'<analysis\s*[">]?\s*(.*?)\s*</analysis>', response, re.DOTALL
            )

            reference_score = 0

            # Process each analysis tag
            for analysis in analysis_tags:
                # Extract issue_type and affected_ids
                issue_type_match = re.search(r"issue_type:\s*([^\n]*)", analysis)
                affected_ids_match = re.search(r"affected_ids:\s*\[(.*?)\]", analysis)

                if not issue_type_match or not affected_ids_match:
                    continue

                issue_type = issue_type_match.group(1).strip()
                affected_ids_str = affected_ids_match.group(1).strip()

                # Parse affected IDs - could be comma-separated list of integers
                try:
                    affected_ids = [
                        int(id.strip()) for id in affected_ids_str.split(",")
                    ]
                except ValueError:
                    affected_ids_match = []

                # Categorize by issue type
                if issue_type == "redundancy_entity" and len(affected_ids) > 0:
                    reference_score += 0.4
                elif issue_type == "redundancy_relationship" and len(affected_ids) > 0:
                    reference_score += 0.4
                elif issue_type == "entity_quality_issue" and len(affected_ids) > 0:
                    reference_score += 0.4
                elif (
                    issue_type == "relationship_quality_issue" and len(affected_ids) > 0
                ):
                    reference_score += 0.4
                elif issue_type == "N/A":
                    reference_score += 0.4

            logger.info(
                f"strict_format_reward_func - format_score: {reference_score}, length: {len(analysis_tags)}"
            )
            if len(analysis_tags) > 0:
                avg_action_score = reference_score / len(analysis_tags)
            else:
                avg_action_score = 0

            this_score += avg_action_score
            # if wandb is enabled, log the reward
            if wandb.run is not None:
                wandb.log(
                    {
                        "format_reward": this_score,
                        "format_avg_action_format_score": avg_action_score,
                        "format_action_count": len(analysis_tags),
                    }
                )
            scores.append(round(this_score, 2))
        except Exception as e:
            logger.error(f"Failed to evaluate response format: {e}")
            scores.append(this_score)

    return scores


def expert_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    ids = kwargs["id"]

    scores = []
    for i, response in enumerate(responses):
        logger.info(f"Compute Expert Reward for {ids[i]}")
        try:
            student_issues = extract_issues(response)
            reference_issues = extract_issues(answer[i])
        except Exception as e:
            logger.error(f"Failed to parse expert_reward_func response {e}")
            scores.append(0)
            continue

        reference_entity_redundancy_issue_count = 0
        reference_relationship_redundancy_issue_count = 0
        reference_entity_quality_issue_count = 0
        reference_relationship_quality_issue_count = 0
        this_score = [0, 0, 0, 0]

        try:
            student_entity_redundancy_issues = normalize_affected_ids(
                [
                    set(student_issue["affected_ids"])
                    for student_issue in student_issues["entity_redundancy_issues"]
                    if len(student_issue["affected_ids"]) > 0
                ]
            )
            reference_entity_redundancy_issues = normalize_affected_ids(
                [
                    set(reference_issue["affected_ids"])
                    for reference_issue in reference_issues["entity_redundancy_issues"]
                    if len(reference_issue["affected_ids"]) > 0
                ]
            )

            if (
                len(student_entity_redundancy_issues) == 0
                and len(reference_entity_redundancy_issues) == 0
            ):
                reference_entity_redundancy_issue_count = 1
                logger.info("entity_redundancy_issues - no issues found, F1 score: 1")
                this_score[0] = 1
            elif (
                len(student_entity_redundancy_issues) > 0
                and len(reference_entity_redundancy_issues) > 0
            ):
                reference_entity_redundancy_issue_count = len(
                    reference_entity_redundancy_issues
                )
                redundancy_entity_f1_scores = []
                for student_ids_set in student_entity_redundancy_issues:
                    for reference_ids_set in reference_entity_redundancy_issues:
                        intersection_result = student_ids_set.intersection(
                            reference_ids_set
                        )
                        if len(intersection_result) > 0:
                            f1_score = compute_f1_score(
                                student_ids_set, reference_ids_set
                            )
                            redundancy_entity_f1_scores.append(f1_score)
                            logger.info(
                                f"entity_redundancy_issues - f1_score: {f1_score}, student_ids_set: {student_ids_set}, reference_ids_set: {reference_ids_set}"
                            )

                if len(reference_entity_redundancy_issues) > 0:
                    total_redundancy_entity_f1_score = sum(redundancy_entity_f1_scores)
                    this_score[0] = round(
                        total_redundancy_entity_f1_score
                        / len(reference_entity_redundancy_issues),
                        2,
                    )
                    logger.info(
                        f"entity_redundancy_issues - total_f1_score: {total_redundancy_entity_f1_score}, average_f1_score: {round(total_redundancy_entity_f1_score / len(reference_entity_redundancy_issues), 2)}"
                    )
                else:
                    logger.info(
                        f"entity_redundancy_issues - no intersection found, F1 score: 0. student answer: {student_entity_redundancy_issues}, reference answer: {reference_entity_redundancy_issues}"
                    )
        except Exception as e:
            logger.error(
                f"Failed to count entity redundancy issues in expert_reward_func response {e}"
            )

        try:
            student_relationship_redundancy_issues = normalize_affected_ids(
                [
                    set(student_issue["affected_ids"])
                    for student_issue in student_issues[
                        "relationship_redundancy_issues"
                    ]
                    if len(student_issue["affected_ids"]) > 0
                ]
            )
            reference_relationship_redundancy_issues = normalize_affected_ids(
                [
                    set(reference_issue["affected_ids"])
                    for reference_issue in reference_issues[
                        "relationship_redundancy_issues"
                    ]
                    if len(reference_issue["affected_ids"]) > 0
                ]
            )

            if (
                len(student_relationship_redundancy_issues) == 0
                and len(reference_relationship_redundancy_issues) == 0
            ):
                reference_relationship_redundancy_issue_count = 1
                logger.info(
                    f"relationship_redundancy_issues - no issues found, F1 score: 1"
                )
                this_score[1] = 1
            elif (
                len(student_relationship_redundancy_issues) > 0
                and len(reference_relationship_redundancy_issues) > 0
            ):
                reference_relationship_redundancy_issue_count = len(
                    reference_relationship_redundancy_issues
                )
                redundancy_relationship_f1_scores = []
                for student_ids_set in student_relationship_redundancy_issues:
                    for reference_ids_set in reference_relationship_redundancy_issues:
                        intersection_result = student_ids_set.intersection(
                            reference_ids_set
                        )
                        if len(intersection_result) > 0:
                            f1_score = compute_f1_score(
                                student_ids_set, reference_ids_set
                            )
                            redundancy_relationship_f1_scores.append(f1_score)
                            logger.info(
                                f"relationship_redundancy_issues - f1_score: {f1_score}, student_ids_set: {student_ids_set}, reference_ids_set: {reference_ids_set}"
                            )
                if len(reference_relationship_redundancy_issues) > 0:
                    total_redundancy_relationship_f1_score = sum(
                        redundancy_relationship_f1_scores
                    )
                    this_score[1] = round(
                        total_redundancy_relationship_f1_score
                        / len(reference_relationship_redundancy_issues),
                        2,
                    )
                    logger.info(
                        f"relationship_redundancy_issues - total_f1_score: {total_redundancy_relationship_f1_score}, average_f1_score: {round(total_redundancy_relationship_f1_score / len(reference_relationship_redundancy_issues), 2)}"
                    )
                else:
                    logger.info(
                        f"relationship_redundancy_issues - no intersection found, F1 score: 0. student answer: {student_relationship_redundancy_issues}, reference answer: {reference_relationship_redundancy_issues}"
                    )
        except Exception as e:
            logger.error(
                f"Failed to count relationship redundancy issues in expert_reward_func response {e}"
            )

        try:
            student_entity_quality_issues_ids = set()
            reference_entity_quality_issues_ids = set()
            for student_issue in student_issues["entity_quality_issues"]:
                student_entity_quality_issues_ids.update(student_issue["affected_ids"])
            for reference_issue in reference_issues["entity_quality_issues"]:
                reference_entity_quality_issues_ids.update(
                    reference_issue["affected_ids"]
                )

            if (
                len(student_entity_quality_issues_ids) == 0
                and len(reference_entity_quality_issues_ids) == 0
            ):
                reference_entity_quality_issue_count = 1
                logger.info("entity_quality_issues - no issues found, F1 score: 1")
                this_score[2] = 1
            elif (
                len(student_entity_quality_issues_ids) > 0
                and len(reference_entity_quality_issues_ids) > 0
            ):
                reference_entity_quality_issue_count = len(
                    reference_entity_quality_issues_ids
                )
                f1_score = compute_f1_score(
                    student_entity_quality_issues_ids,
                    reference_entity_quality_issues_ids,
                )
                this_score[2] = round(f1_score, 2)
                logger.info(
                    f"entity_quality_issues - f1_score: {f1_score}, student_ids_set: {student_entity_quality_issues_ids}, reference_ids_set: {reference_entity_quality_issues_ids}"
                )
        except Exception as e:
            logger.error(
                f"Failed to count entity quality issues in expert_reward_func response {e}"
            )

        try:
            student_relationship_quality_issues_ids = set()
            reference_relationship_quality_issues_ids = set()
            for student_issue in student_issues["relationship_quality_issues"]:
                student_relationship_quality_issues_ids.update(
                    student_issue["affected_ids"]
                )
            for reference_issue in reference_issues["relationship_quality_issues"]:
                reference_relationship_quality_issues_ids.update(
                    reference_issue["affected_ids"]
                )

            if (
                len(student_relationship_quality_issues_ids) == 0
                and len(reference_relationship_quality_issues_ids) == 0
            ):
                logger.info(
                    "relationship_quality_issues - no issues found, F1 score: 1"
                )
                reference_relationship_quality_issue_count = 1
                this_score[3] = 1
            elif (
                len(student_relationship_quality_issues_ids) > 0
                and len(reference_relationship_quality_issues_ids) > 0
            ):
                reference_relationship_quality_issue_count = len(
                    reference_relationship_quality_issues_ids
                )
                f1_score = compute_f1_score(
                    student_relationship_quality_issues_ids,
                    reference_relationship_quality_issues_ids,
                )
                this_score[3] = round(f1_score, 2)
                logger.info(
                    f"relationship_quality_issues - f1_score: {f1_score}, student_ids_set: {student_relationship_quality_issues_ids}, reference_ids_set: {reference_relationship_quality_issues_ids}"
                )
        except Exception as e:
            logger.error(
                f"Failed to count relationship quality issues in expert_reward_func response {e}"
            )

        total_issue_count = (
            reference_entity_redundancy_issue_count
            + reference_relationship_redundancy_issue_count
            + reference_entity_quality_issue_count
            + reference_relationship_quality_issue_count
        )
        weightd_score = (
            (reference_entity_redundancy_issue_count / total_issue_count)
            * this_score[0]
            + (reference_relationship_redundancy_issue_count / total_issue_count)
            * this_score[1]
            + (reference_entity_quality_issue_count / total_issue_count) * this_score[2]
            + (reference_relationship_quality_issue_count / total_issue_count)
            * this_score[3]
        )
        scores.append(round(weightd_score, 2))

    return scores


def problem_analysis_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function for the Problem Analysis stage (30% weight in cognitive model).

    Uses LLM to evaluate how well the model analyzes problems in the graph.
    """
    responses = [completion[0]["content"] for completion in completions]
    graphs = kwargs["graph"]
    valid_analysis_count = kwargs["valid_analysis_count"]
    ids = kwargs["id"]
    scores = []

    for i, response in enumerate(responses):
        logger.info(f"Compute Problem Analysis Reward for {ids[i]}")
        try:
            analysis_tags = re.findall(
                r'<analysis\s*[">]?\s*(.*?)\s*</analysis>', response, re.DOTALL
            )

            valid_analysis = []
            # Process each analysis tag
            for analysis in analysis_tags:
                # Extract issue_type and affected_ids
                issue_type_match = re.search(r"issue_type:\s*([^\n]*)", analysis)
                affected_ids_match = re.search(r"affected_ids:\s*\[(.*?)\]", analysis)

                if not issue_type_match or not affected_ids_match:
                    continue

                issue_type = issue_type_match.group(1).strip()
                affected_ids_str = affected_ids_match.group(1).strip()

                # Parse affected IDs - could be comma-separated list of integers
                try:
                    affected_ids = [
                        int(id.strip()) for id in affected_ids_str.split(",")
                    ]
                except ValueError:
                    affected_ids_match = []

                # Categorize by issue type
                if issue_type == "redundancy_entity" and len(affected_ids) > 0:
                    valid_analysis.append(analysis)
                elif issue_type == "redundancy_relationship" and len(affected_ids) > 0:
                    valid_analysis.append(analysis)
                elif issue_type == "entity_quality_issue" and len(affected_ids) > 0:
                    valid_analysis.append(analysis)
                elif (
                    issue_type == "relationship_quality_issue" and len(affected_ids) > 0
                ):
                    valid_analysis.append(analysis)
                elif issue_type == "N/A":
                    valid_analysis.append(analysis)

            if len(valid_analysis) == 0:
                logger.error(
                    f"problem_analysis_reward_func - No valid analysis found for response: {response}"
                )
                scores.append(0.0)
                continue

            analysis_content = "</analysis>\n\n<analysis>".join(valid_analysis)
            analysis_content = "<analysis>\n" + analysis_content + "\n</analysis>"

            # Prepare evaluation prompt with graph and response
            evaluation_prompt = PROBLEM_ANALYSIS_EVALUATION_PROMPT.format(
                graph=graphs[i], response=analysis_content, reference_answer=answer[i]
            )

            try:
                # Get evaluation from LLM
                logger.info(
                    f"problem_analysis_reward_func - prompt: {evaluation_prompt}"
                )
                evaluation_response = llm_client.generate(evaluation_prompt)
            except Exception as e:
                logger.error(f"Failed to evaluate problem analysis: {e}")
                evaluation_response = llm_client_bk.generate(evaluation_prompt)

            logger.info(
                f"problem_analysis_reward_func - response: {evaluation_response}"
            )

            # Extract and parse JSON
            evaluation_json = extract_json(evaluation_response)
            if not evaluation_json:
                logger.error(
                    f"problem_analysis_reward_func - No evaluation JSON found for response: {response}"
                )
                scores.append(0.0)
                continue

            evaluation_results = json.loads(evaluation_json)
            analysis_quality_score = min(
                evaluation_results.get("total_analysis_quality_score", 0)
                / valid_analysis_count[i],
                1.0,
            )

            if wandb.run is not None:
                wandb.log({"analysis_quality_score": analysis_quality_score})
            scores.append(round(analysis_quality_score, 2))

        except Exception as e:
            logger.error(f"Failed to evaluate problem analysis: {e}")
            scores.append(0.0)

    return scores
