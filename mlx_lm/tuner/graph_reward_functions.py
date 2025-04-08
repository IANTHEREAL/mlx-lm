import re
from typing import Callable, List, Optional


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
    issue_tags = re.findall(
        r'<issue\s*[">]?\s*(.*?)\s*</issue>', response, re.DOTALL
    )

    entity_redundancy_issues = []
    relationship_redundancy_issues = []
    entity_quality_issues = []
    relationship_quality_issues = []

    # Process each analysis tag
    for issue in issue_tags:
        # Extract issue_type and affected_ids
        issue_type_match = re.search(r"issue_type:\s*([^\n]*)", issue)
        affected_ids_match = re.search(r"affected_ids:\s*\[(.*?)\]", issue)
        reasoning_match = re.search(r"reasoning:\s*([\s\S]*?)\n", issue)
        confidence_match = re.search(r"confidence:\s*([\w_/]+)\n", issue)
        facto_search_match = re.search(r"facto_search:\s*([\s\S]*?)\n", issue)

        if (
            not issue_type_match
            or not affected_ids_match
            or not reasoning_match
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
        confidence = confidence_match.group(1).strip()
        if facto_search_match:
            facto_search = facto_search_match.group(1).strip()
        else:
            facto_search = ""

        issue = {
            "issue_type": issue_type,
            "affected_ids": affected_ids,
            "reasoning": reasoning,
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


def reward_len(completions, answer: list, **kwargs):
    responses = [completion for completion in completions]
    scores = []
    # print(f"reference answer:\n {answer[0]}")
    for index, response in enumerate(responses):
        # print(f"answer_{index}, {response}")
        completion_issues = re.findall(
            r'<issue\s*[">]?\s*(.*?)\s*</issue>', response, re.DOTALL
        )
        reference_issues = re.findall(
            r'<issue\s*[">]?\s*(.*?)\s*</issue>', answer[index], re.DOTALL
        )

        if len(reference_issues) == 0:
            scores.append(len(completion_issues) == 0)
        else:
            scores.append(1-abs(1 - len(completion_issues)/len(reference_issues)))
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

            answer_sections = response.split("</think>")
            if len(answer_sections) != 2:
                scores.append(this_score)
                continue

            issue_start_count = response.count("<issue>")
            issue_end_count = response.count("</issue>")
            if issue_start_count != issue_end_count:
                this_score -= abs(issue_start_count - issue_end_count) * 0.1

            issue_tags = re.findall(
                r'<issue\s*[">]?\s*(.*?)\s*</issue>', response, re.DOTALL
            )

            reference_score = 0

            # Process each analysis tag
            for issue in issue_tags:
                # Extract issue_type and affected_ids
                issue_type_match = re.search(r"issue_type:\s*([^\n]*)", issue)
                affected_ids_match = re.search(r"affected_ids:\s*\[(.*?)\]", issue)

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

            if len(issue_tags) > 0:
                avg_action_score = reference_score / len(issue_tags)
            else:
                avg_action_score = 0

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
            print(f"Failed to parse expert_reward_func response {e}")
            scores.append(0)
            continue

        entity_redundancy_issue_count = 0
        relationship_redundancy_issue_count = 0
        entity_quality_issue_count = 0
        relationship_quality_issue_count = 0
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
                entity_redundancy_issue_count = 1
                this_score[0] = 1
            elif (
                len(student_entity_redundancy_issues) > 0
                and len(reference_entity_redundancy_issues) > 0
            ):
                entity_redundancy_issue_count = max(
                    len(reference_entity_redundancy_issues),
                    len(student_entity_redundancy_issues)
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

                if len(reference_entity_redundancy_issues) > 0:
                    total_redundancy_entity_f1_score = sum(redundancy_entity_f1_scores)
                    this_score[0] = round(
                        total_redundancy_entity_f1_score
                        / len(reference_entity_redundancy_issues),
                        2,
                    )
                else:
                    entity_redundancy_issue_count = max(
                        len(reference_entity_redundancy_issues),
                        len(student_entity_redundancy_issues)
                    )
        except Exception as e:
            print(
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
                relationship_redundancy_issue_count = 1
                this_score[1] = 1
            elif (
                len(student_relationship_redundancy_issues) > 0
                and len(reference_relationship_redundancy_issues) > 0
            ):
                relationship_redundancy_issue_count = max(
                    len(reference_relationship_redundancy_issues),
                    len(student_relationship_redundancy_issues)
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
                if len(reference_relationship_redundancy_issues) > 0:
                    total_redundancy_relationship_f1_score = sum(
                        redundancy_relationship_f1_scores
                    )
                    this_score[1] = round(
                        total_redundancy_relationship_f1_score
                        / len(reference_relationship_redundancy_issues),
                        2,
                    )
                else:
                    relationship_redundancy_issue_count = max(
                        len(reference_relationship_redundancy_issues),
                        len(student_relationship_redundancy_issues)
                    )
        except Exception as e:
            print(
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
                entity_quality_issue_count = 1
                this_score[2] = 1
            elif (
                len(student_entity_quality_issues_ids) > 0
                and len(reference_entity_quality_issues_ids) > 0
            ):
                entity_quality_issue_count = max(
                    len(reference_entity_quality_issues_ids),
                    len(student_entity_quality_issues_ids)
                )
                f1_score = compute_f1_score(
                    student_entity_quality_issues_ids,
                    reference_entity_quality_issues_ids,
                )
                this_score[2] = round(f1_score, 2)
            else:
                entity_quality_issue_count = max(
                    len(reference_entity_quality_issues_ids),
                    len(student_entity_quality_issues_ids)
                )
        except Exception as e:
            print(
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
                relationship_quality_issue_count = 1
                this_score[3] = 1
            elif (
                len(student_relationship_quality_issues_ids) > 0
                and len(reference_relationship_quality_issues_ids) > 0
            ):
                relationship_quality_issue_count = max(
                    len(reference_relationship_quality_issues_ids),
                    len(student_relationship_quality_issues_ids)
                )
                f1_score = compute_f1_score(
                    student_relationship_quality_issues_ids,
                    reference_relationship_quality_issues_ids,
                )
                this_score[3] = round(f1_score, 2)
            else:
                relationship_quality_issue_count = max(
                    len(reference_relationship_quality_issues_ids),
                    len(student_relationship_quality_issues_ids)
                )
        except Exception as e:
            print(
                f"Failed to count relationship quality issues in expert_reward_func response {e}"
            )

        total_issue_count = (
            entity_redundancy_issue_count
            + relationship_redundancy_issue_count
            + entity_quality_issue_count
            + relationship_quality_issue_count
        )
        if total_issue_count == 0:
            print(
                f"expert_reward_func - no issues found, F1 score: 0, student answer: {student_issues}, reference answer: {reference_issues}"
            )
            scores.append(0)
            continue
        weightd_score = (
            (entity_redundancy_issue_count / total_issue_count)
            * this_score[0]
            + (relationship_redundancy_issue_count / total_issue_count)
            * this_score[1]
            + (entity_quality_issue_count / total_issue_count) * this_score[2]
            + (relationship_quality_issue_count / total_issue_count)
            * this_score[3]
        )
        scores.append(round(weightd_score, 2))

    return scores
