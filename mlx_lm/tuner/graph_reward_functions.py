import re
import math
import json
from typing import Callable, List, Optional


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


def extract_issues(response: str):
    response_json_str = extract_json(response)
    if not response_json_str:
        return {
            "entity_redundancy_issues": [],
            "relationship_redundancy_issues": [],
            "entity_quality_issues": [],
            "relationship_quality_issues": [],
            "missing_relationship_issues": [],
        }
    issue_tags = json.loads(response_json_str)

    entity_redundancy_issues = []
    relationship_redundancy_issues = []
    entity_quality_issues = []
    relationship_quality_issues = []
    missing_relationship_issues = []

    # Process each analysis tag
    for analysis in issue_tags:
        # Extract issue_type and affected_ids
        issue_type = analysis.get("issue_type", None)
        affected_ids = analysis.get("affected_ids", [])
        reasoning = analysis.get("reasoning", None)
        confidence = analysis.get("confidence", None)

        if not issue_type or not affected_ids or not reasoning or not confidence:
            continue

        issue = {
            "issue_type": issue_type,
            "affected_ids": affected_ids,
            "reasoning": reasoning,
            "confidence": confidence,
            "facto_search": "",
        }
        # Categorize by issue type
        if issue_type == "redundancy_entity" and len(affected_ids) >= 2:
            entity_redundancy_issues.append(issue)
        elif issue_type == "redundancy_relationship" and len(affected_ids) >= 2:
            relationship_redundancy_issues.append(issue)
        elif issue_type == "entity_quality_issue" and len(affected_ids) > 0:
            entity_quality_issues.append(issue)
        elif issue_type == "relationship_quality_issue" and len(affected_ids) > 0:
            relationship_quality_issues.append(issue)
        elif issue_type == "missing_relationship" and len(affected_ids) == 2:
            missing_relationship_issues.append(issue)

    return {
        "entity_redundancy_issues": entity_redundancy_issues,
        "relationship_redundancy_issues": relationship_redundancy_issues,
        "entity_quality_issues": entity_quality_issues,
        "relationship_quality_issues": relationship_quality_issues,
        "missing_relationship_issues": missing_relationship_issues,
    }


def reward_len(completions, answer: list, **kwargs):
    responses = [completion for completion in completions]
    scores = []
    # print(f"reference answer:\n {answer[0]}")
    for index, response in enumerate(responses):
        completion_issues = extract_issues(response)
        reference_issues = extract_issues(answer[index])

        num_completion_issues = sum(
            [len(issue_list) for issue_list in completion_issues.values()]
        )
        num_reference_issues = sum(
            [len(issue_list) for issue_list in reference_issues.values()]
        )

        if num_reference_issues == 0:
            # If there are no reference issues, reward 1 for no completion issues, 0 otherwise.
            score = 1.0 if num_completion_issues == 0 else 0.0
        else:
            diff_ratio = (
                abs(num_completion_issues - num_reference_issues) / num_reference_issues
            )
            score = math.exp(-4 * (diff_ratio**2))

        scores.append(round(score, 2))

    return scores


def strict_format_reward_func(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    """Reward function that checks if the response contains valid graph optimization actions."""
    responses = [completion for completion in completions]
    scores = []

    for i, response in enumerate(responses):
        this_score = 0
        try:
            this_score += (
                2
                - abs(response.count("<think>") - 1)
                - abs(response.count("</think>") - 1)
            ) * 0.2

            response_json_str = extract_json(response)
            if response_json_str is None:
                scores.append(this_score)
                continue

            analysis_tags = json.loads(response_json_str)

            if len(analysis_tags) > 0:
                reference_score = 0
                # Process each analysis tag
                for analysis in analysis_tags:
                    issue_type = analysis.get("issue_type", None)
                    affected_ids = analysis.get("affected_ids", [])

                    # Categorize by issue type
                    if issue_type == "redundancy_entity" and len(affected_ids) >= 2:
                        reference_score += 0.4
                    elif (
                        issue_type == "redundancy_relationship"
                        and len(affected_ids) >= 2
                    ):
                        reference_score += 0.4
                    elif issue_type == "entity_quality_issue" and len(affected_ids) > 0:
                        reference_score += 0.4
                    elif (
                        issue_type == "relationship_quality_issue"
                        and len(affected_ids) > 0
                    ):
                        reference_score += 0.4
                    elif (
                        issue_type == "missing_relationship" and len(affected_ids) == 2
                    ):
                        reference_score += 0.4

                if len(analysis_tags) > 0:
                    avg_action_score = reference_score / len(analysis_tags)
                else:
                    avg_action_score = 0
            else:
                avg_action_score = 0.4

            this_score += avg_action_score
            scores.append(round(this_score, 2))
        except Exception as e:
            scores.append(this_score)

    return scores


def expert_reward_func(
    prompts: list, completions: list, answer: list, types: Optional[list] = None
) -> list[float]:
    responses = [completion for completion in completions]
    scores = []

    for i, response in enumerate(responses):
        try:
            student_issues = extract_issues(response)
            reference_issues = extract_issues(answer[i])
        except Exception as e:
            print(f"Failed to parse expert_reward_func response {e}, {response}")
            scores.append(0)
            continue

        total_reference_issue_count = 0
        for issue_list in reference_issues.values():
            for issue in issue_list:
                if float(issue["confidence"]) >= 0.9:
                    total_reference_issue_count += 1

        total_student_issue_count = sum(
            [len(issues) for issues in student_issues.values()]
        )

        if total_reference_issue_count == 0 and total_student_issue_count == 0:
            scores.append(1)
            continue
        elif total_reference_issue_count == 0 or total_student_issue_count == 0:
            for issues in student_issues.values():
                for issue in issues:
                    print(f"<invalid_issue>{issue}</invalid_issue>")
            scores.append(0)
            continue

        student_score = 0
        reference_score = 0

        try:
            high_confidence_entity_redundancy_issues = [
                issue
                for issue in reference_issues["entity_redundancy_issues"]
                if float(issue["confidence"]) >= 1.8
            ]
            moderate_confidence_entity_redundancy_issues = [
                issue
                for issue in reference_issues["entity_redundancy_issues"]
                if float(issue["confidence"]) >= 0.9
                and float(issue["confidence"]) < 1.8
            ]

            student_entity_redundancy_issues = normalize_affected_ids(
                [
                    set(student_issue["affected_ids"])
                    for student_issue in student_issues["entity_redundancy_issues"]
                    if len(student_issue["affected_ids"]) > 0
                ]
            )
            high_confidence_reference_entity_redundancy_issues = normalize_affected_ids(
                [
                    set(reference_issue["affected_ids"])
                    for reference_issue in high_confidence_entity_redundancy_issues
                    if len(reference_issue["affected_ids"]) > 0
                ]
            )
            moderate_confidence_reference_entity_redundancy_issues = normalize_affected_ids(
                [
                    set(reference_issue["affected_ids"])
                    for reference_issue in moderate_confidence_entity_redundancy_issues
                    if len(reference_issue["affected_ids"]) > 0
                ]
            )
            reference_score += (
                len(high_confidence_reference_entity_redundancy_issues)
                + len(moderate_confidence_reference_entity_redundancy_issues) * 0.5
            )

            if len(student_entity_redundancy_issues) > 0:
                for student_ids_set in student_entity_redundancy_issues:
                    this_score = 0
                    for (
                        reference_ids_set
                    ) in high_confidence_reference_entity_redundancy_issues:
                        intersection_result = student_ids_set.intersection(
                            reference_ids_set
                        )
                        if len(intersection_result) > 0:
                            f1_score = compute_f1_score(
                                student_ids_set, reference_ids_set
                            )
                            this_score += f1_score

                    for (
                        reference_ids_set
                    ) in moderate_confidence_reference_entity_redundancy_issues:
                        intersection_result = student_ids_set.intersection(
                            reference_ids_set
                        )
                        if len(intersection_result) > 0:
                            f1_score = compute_f1_score(
                                student_ids_set, reference_ids_set
                            )
                            this_score += f1_score * 0.5

                    student_score += this_score

                    if this_score == 0:
                        student_score -= 0.5
                        print(
                            f"<invalid_entity_redundancy_issues>{student_ids_set}</invalid_entity_redundancy_issues>"
                        )

        except Exception as e:
            print(
                f"Failed to count entity redundancy issues in expert_reward_func response {e}"
            )

        try:
            high_confidence_relationship_redundancy_issues = [
                issue
                for issue in reference_issues["relationship_redundancy_issues"]
                if float(issue["confidence"]) >= 1.8
            ]
            moderate_confidence_relationship_redundancy_issues = [
                issue
                for issue in reference_issues["relationship_redundancy_issues"]
                if float(issue["confidence"]) >= 0.9
                and float(issue["confidence"]) < 1.8
            ]

            student_relationship_redundancy_issues = normalize_affected_ids(
                [
                    set(student_issue["affected_ids"])
                    for student_issue in student_issues[
                        "relationship_redundancy_issues"
                    ]
                    if len(student_issue["affected_ids"]) > 0
                ]
            )
            high_confidence_reference_relationship_redundancy_issues = normalize_affected_ids(
                [
                    set(reference_issue["affected_ids"])
                    for reference_issue in high_confidence_relationship_redundancy_issues
                    if len(reference_issue["affected_ids"]) > 0
                ]
            )

            moderate_confidence_reference_relationship_redundancy_issues = normalize_affected_ids(
                [
                    set(reference_issue["affected_ids"])
                    for reference_issue in moderate_confidence_relationship_redundancy_issues
                    if len(reference_issue["affected_ids"]) > 0
                ]
            )

            reference_score += (
                len(high_confidence_reference_relationship_redundancy_issues)
                + len(moderate_confidence_reference_relationship_redundancy_issues)
                * 0.5
            )

            if len(student_relationship_redundancy_issues) > 0:
                for student_ids_set in student_relationship_redundancy_issues:
                    this_score = 0
                    for (
                        reference_ids_set
                    ) in high_confidence_reference_relationship_redundancy_issues:
                        intersection_result = student_ids_set.intersection(
                            reference_ids_set
                        )
                        if len(intersection_result) > 0:
                            f1_score = compute_f1_score(
                                student_ids_set, reference_ids_set
                            )
                            this_score += f1_score

                    for (
                        reference_ids_set
                    ) in moderate_confidence_reference_relationship_redundancy_issues:
                        intersection_result = student_ids_set.intersection(
                            reference_ids_set
                        )
                        if len(intersection_result) > 0:
                            f1_score = compute_f1_score(
                                student_ids_set, reference_ids_set
                            )
                            this_score += f1_score * 0.5

                    student_score += this_score

                    if this_score == 0:
                        student_score -= 0.5
                        print(
                            f"<invalid_relationship_redundancy_issues>{student_ids_set}</invalid_relationship_redundancy_issues>"
                        )

        except Exception as e:
            print(
                f"Failed to count relationship redundancy issues in expert_reward_func response {e}"
            )

        try:
            student_entity_quality_issues_ids = set()
            high_confidence_reference_entity_quality_issues_ids = set()
            moderate_confidence_reference_entity_quality_issues_ids = set()

            for student_issue in student_issues["entity_quality_issues"]:
                student_entity_quality_issues_ids.update(student_issue["affected_ids"])

            for reference_issue in reference_issues["entity_quality_issues"]:
                if float(reference_issue["confidence"]) >= 1.8:
                    high_confidence_reference_entity_quality_issues_ids.update(
                        reference_issue["affected_ids"]
                    )
                elif (
                    float(reference_issue["confidence"]) >= 0.9
                    and float(reference_issue["confidence"]) < 1.8
                ):
                    moderate_confidence_reference_entity_quality_issues_ids.update(
                        reference_issue["affected_ids"]
                    )

            reference_score += (
                len(high_confidence_reference_entity_quality_issues_ids)
                + len(moderate_confidence_reference_entity_quality_issues_ids) * 0.5
            )

            if len(student_entity_quality_issues_ids) > 0:
                for student_id in student_entity_quality_issues_ids:
                    this_score = 0
                    if (
                        student_id
                        in high_confidence_reference_entity_quality_issues_ids
                    ):
                        this_score += 1
                    if (
                        student_id
                        in moderate_confidence_reference_entity_quality_issues_ids
                    ):
                        this_score += 0.5

                    student_score += this_score

                    if this_score == 0:
                        student_score -= 0.5
                        print(
                            f"<invalid_entity_quality_issues>{student_id}</invalid_entity_quality_issues>"
                        )

        except Exception as e:
            print(
                f"Failed to count entity quality issues in expert_reward_func response {e}"
            )

        try:
            student_relationship_quality_issues_ids = set()
            high_confidence_reference_relationship_quality_issues_ids = set()
            moderate_confidence_reference_relationship_quality_issues_ids = set()

            for student_issue in student_issues["relationship_quality_issues"]:
                student_relationship_quality_issues_ids.update(
                    student_issue["affected_ids"]
                )
            for reference_issue in reference_issues["relationship_quality_issues"]:
                if float(reference_issue["confidence"]) >= 1.8:
                    high_confidence_reference_relationship_quality_issues_ids.update(
                        reference_issue["affected_ids"]
                    )
                elif (
                    float(reference_issue["confidence"]) >= 0.9
                    and float(reference_issue["confidence"]) < 1.8
                ):
                    moderate_confidence_reference_relationship_quality_issues_ids.update(
                        reference_issue["affected_ids"]
                    )

            reference_score += (
                len(high_confidence_reference_relationship_quality_issues_ids)
                + len(moderate_confidence_reference_relationship_quality_issues_ids)
                * 0.5
            )

            if len(student_relationship_quality_issues_ids) > 0:
                for student_id in student_relationship_quality_issues_ids:
                    this_score = 0
                    if (
                        student_id
                        in high_confidence_reference_relationship_quality_issues_ids
                    ):
                        this_score += 1
                    if (
                        student_id
                        in moderate_confidence_reference_relationship_quality_issues_ids
                    ):
                        this_score += 0.5

                    student_score += this_score
                    if this_score == 0:
                        student_score -= 0.5
                        print(
                            f"<invalid_relationship_quality_issues>{student_id}</invalid_relationship_quality_issues>"
                        )

        except Exception as e:
            print(
                f"Failed to count relationship quality issues in expert_reward_func response {e}"
            )

        if reference_score == 0:
            if student_score == 0:
                scores.append(1)
            else:
                scores.append(0)
        else:
            scores.append(
                min(
                    round(
                        student_score / reference_score,
                        2,
                    ),
                    1,
                )
            )

    return scores
