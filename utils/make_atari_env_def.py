import inspect
from stable_baselines3.common.env_util import make_atari_env
print(inspect.getsource(make_atari_env))

# def make_atari_env(
#     env_id: Union[str, Callable[..., gym.Env]],
#     n_envs: int = 1,
#     seed: Optional[int] = None,
#     start_index: int = 0,
#     monitor_dir: Optional[str] = None,
#     wrapper_kwargs: Optional[dict[str, Any]] = None,
#     env_kwargs: Optional[dict[str, Any]] = None,
#     vec_env_cls: Optional[Union[type[DummyVecEnv], type[SubprocVecEnv]]] = None,
#     vec_env_kwargs: Optional[dict[str, Any]] = None,
#     monitor_kwargs: Optional[dict[str, Any]] = None,
# ) -> VecEnv:
#     """
#     Create a wrapped, monitored VecEnv for Atari.
#     It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

#     .. note::
#         By default, the ``AtariWrapper`` uses ``terminal_on_life_loss=True``, which causes
#         ``env.reset()`` to perform a no-op step instead of truly resetting when the environment
#         terminates due to a loss of life (but not game over). To ensure ``reset()`` always
#         resets the env, pass ``wrapper_kwargs=dict(terminal_on_life_loss=False)``.

#     :param env_id: either the env ID, the env class or a callable returning an env
#     :param n_envs: the number of environments you wish to have in parallel
#     :param seed: the initial seed for the random number generator
#     :param start_index: start rank index
#     :param monitor_dir: Path to a folder where the monitor files will be saved.
#         If None, no file will be written, however, the env will still be wrapped
#         in a Monitor wrapper to provide additional information about training.
#     :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
#     :param env_kwargs: Optional keyword argument to pass to the env constructor
#     :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
#     :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
#     :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
#     :return: The wrapped environment
#     """
#     return make_vec_env(
#         env_id,
#         n_envs=n_envs,
#         seed=seed,
#         start_index=start_index,
#         monitor_dir=monitor_dir,
#         wrapper_class=AtariWrapper,
#         env_kwargs=env_kwargs,
#         vec_env_cls=vec_env_cls,
#         vec_env_kwargs=vec_env_kwargs,
#         monitor_kwargs=monitor_kwargs,
#         wrapper_kwargs=wrapper_kwargs,
#     )
