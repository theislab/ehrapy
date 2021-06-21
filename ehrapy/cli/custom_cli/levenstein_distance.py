from typing import Tuple

SIMILARITY_USE_FACTOR = 1 / 3
SIMILARITY_SUGGEST_FACTOR = 2 / 3


def levensthein_dist(input_command: str, candidate: str) -> int:
    """Implementation of the Levensthein algorithm.

    Implementation of the Levenshtein distance algorithm to determine, in case of a non-existing command,
    whether there is a similar command to suggest.

    Args:
        input_command: The non-existing command the user provided as input
        candidate: The (possible similar) alternative command

    Returns:
        The similarity between the two strings measured by the levensthein distance
    """

    if not input_command or not candidate:
        return max(len(input_command), len(candidate))  # at least one string is empty

    dp_table = [[0 for col in range(len(input_command) + 1)] for row in range(len(candidate) + 1)]

    dp_table[0] = list(range(0, len(input_command) + 1))
    for i in range(1, len(candidate) + 1):
        dp_table[i][0] = i

    # now choose minimum levensthein distance from the three option delete/replace/insert
    # if chars are the same -> levensthein distance is the same as for those substring
    # without these chars of input_command and candidate

    for i in range(1, len(candidate) + 1):
        for j in range(1, len(input_command) + 1):
            # choose minimum edit distance from delete, replace or insert at current substring
            if input_command[j - 1] == candidate[i - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1]
            else:
                dp_table[i][j] = min(min(dp_table[i][j - 1], dp_table[i - 1][j - 1]), dp_table[i - 1][j]) + 1

    return dp_table[len(candidate)][len(input_command)]


def most_similar_command(command: str, command_list: set) -> Tuple[list, str]:  # pragma: no cover
    """Determine whether its possible to suggest a similar command.

    The similarity is determined by the levensthein distance and a factor (currently 1/3)
    sets a limit where a similar command is useful to be automatically used.
    If the difference diff is 1/3 < diff <= 2/3, one or more similar commands could be suggested,
    but not used automatically.

    Args:
        command: The command provided by the user
        command_list: The commands that are available by the users specific action

    Returns:
        A list of similar command(s) or the empty string if there's none and
         a string that indicates the action to be taken
    """
    min_use = 999999  # some random large integer -> we will never have handles that are larger than 1000 character
    min_suggest = 999999
    sim_command_use = []
    sim_command_suggest = []

    # for each valid handle calculate the levensthein distance and if one is found that is a new minimal distance,
    # replace it and take this handle as the most similar command.
    for handle in command_list:
        dist = levensthein_dist(command, handle)

        # the more restrict condition for automatic use
        lim_use = int(len(command) * SIMILARITY_USE_FACTOR)

        # the weaker condition for command suggestion
        lim_suggest = int(len(command) * SIMILARITY_SUGGEST_FACTOR)

        # check if the command is close to the inputted command so it can be automatically used
        if lim_use >= dist:
            if min_use > dist:  # and min >= dist:
                min_use = dist
                sim_command_use = [handle]
            elif min_use == dist:
                sim_command_use.append(handle)

        # the input is not very close to any command, but maybe a similar one can be suggested?
        elif lim_use < dist <= lim_suggest:
            if min_suggest > dist:  # and min >= dist:
                min_suggest = dist
                sim_command_suggest = [handle]
            elif min_suggest == dist:
                sim_command_suggest.append(handle)

    # return the use list, as those are closer, but if its empty, return the list of suggested commands
    # (or if that is empty too, an empty list)
    return (
        (sim_command_use, "use")
        if sim_command_use
        else (sim_command_suggest, "suggest")
        if sim_command_suggest
        else ([], "")
    )
