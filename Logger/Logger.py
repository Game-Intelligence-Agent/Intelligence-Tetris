from statistics import mean, median

class Logger:

    @staticmethod
    def log_result(tb, name, scores, log_every, episode):

        avg_score = mean(scores[-log_every:])
        min_score = min(scores[-log_every:])
        max_score = max(scores[-log_every:])

        tb.add_scalar(avg_score, episode, f'Avg {name} every {log_every} Episode')
        tb.add_scalar(max_score, episode, f'Max {name} every {log_every} Episode')
        tb.add_scalar(min_score, episode, f'Min {name} every {log_every} Episode')

        print(f'[{name}s For Last {log_every} Games] Avg {name}: {avg_score}, Max {name}: {max_score}, Min {name}: {min_score}')

        return max_score, min_score, avg_score
