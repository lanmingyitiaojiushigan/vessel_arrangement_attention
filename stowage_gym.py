import warnings
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional
from enum import Enum
import colorsys


class StateIds(Enum):
    BAY = 0
    ROW = 1
    TIER = 2
    IS_OCCUPIED = 3
    GROUP = 4


class StowageEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(self, config: Dict = None, render_mode: Optional[str] = None):
        if config is None:
            config = {}

        self.vessel_shape = config.get("vessel_shape", (1, 2, 2))
        self.yard_shape = config.get("yard_shape", (2, 2, 2))
        self.num_containers = config.get("num_containers", 4)
        self.group_num = config.get("group_num", 1)
        self.group_placement = config.get("group_placement", "fixed")
        self.seed = config.get("seed", 0)
        self.action_mask = config.get("action_mask", "default")
        # Calculate total physical slots
        self.total_shifters = 0
        self.total_vessel_slots = self.vessel_shape[0] * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_slots = self.yard_shape[0] * self.yard_shape[1] * self.yard_shape[2]
        self.total_timesteps = 0  # Use to deal with timeout

        self.num_containers = min(self.num_containers, self.total_yard_slots)
        if self.num_containers > self.total_yard_slots:
            warnings.warn(
                f"Number of containers is set to {self.num_containers} as it exceeds the total yard slots {self.total_yard_slots}"
            )
        self.num_slot_attrs = 5

        # Calculate total coordinates
        self.num_vessel_bay = (
                self.vessel_shape[0] // 2 + self.vessel_shape[0]
        )  # number of bay coords. e.g., need 6 coords to encode 4 bays
        self.num_yard_bay = self.yard_shape[0] // 2 + self.yard_shape[0]
        self.total_vessel_coords = self.num_vessel_bay * self.vessel_shape[1] * self.vessel_shape[2]
        self.total_yard_coords = self.num_yard_bay * self.yard_shape[1] * self.yard_shape[2]
        self.obs_coords = self.total_vessel_coords + self.total_yard_coords + 1  # +1 for current target

        # Crane
        self.num_cranes = min(config.get("num_cranes", 2), self.vessel_shape[0])
        self.time_penalty_coef = config.get("time_penalty_coef", 0.01)

        # Sequencer
        self.current_vessel_slot = None
        self.vessel_slots_filled = 0

        self.crane_busy_until = np.zeros(self.num_cranes)
        self.current_time = 0

        observation_space = gym.spaces.Box(
            low=0,
            high=max(
                max(self.num_vessel_bay, self.num_yard_bay),  # bay upper limit
                max(self.vessel_shape[1], self.yard_shape[1]),  # row upper limit
                max(self.vessel_shape[2], self.yard_shape[2]),  # tier upper limit
                1,  # occupied upper limit
                self.group_num,  # group upper limit
            ),
            shape=(self.obs_coords * self.num_slot_attrs,),
            dtype=np.int64,
        )

        if self.action_mask == "default":
            self.observation_space = observation_space
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": observation_space,
                    "mask": gym.spaces.Box(low=0, high=1, shape=(self.total_yard_coords,), dtype=np.bool_),
                }
            )

        self.action_space = gym.spaces.Discrete(self.total_yard_coords)

        # Render part
        self.render_mode = render_mode
        self.screen_width = (
                35 * max(self.vessel_shape[1] * self.vessel_shape[0], self.yard_shape[1] * self.yard_shape[0]) + 60
        )
        self.screen_height = 60 * max(self.vessel_shape[2], self.yard_shape[2]) + 150
        self.screen = None

    def step(self, action):
        self.total_timesteps += 1
        truncated = False if self.total_timesteps < self.num_containers * 10 else True  # Handle timeout
        truncated = False
        info = {}
        valid_actions = self._get_valid_yard_actions()
        if type(valid_actions) is not list:
            valid_actions = valid_actions.tolist()

        # print(valid_actions)

        # if action not in valid_actions:    # 这个在随机策略中没用到
        #     reward = -100.0
        #     observation = self._create_observation()
        #     info["yard_mask"] = valid_actions   # 这是什么意思
        #     terminated = False
        #     return observation, reward, terminated, truncated, info

        shifters = self._process_shifters(action, self.current_vessel_slot)  # action的解空间其实有300个，但是只有200个集装箱，正常
        # Sequencer select next vessel slot
        self.current_vessel_slot = self._get_next_vessel_slot()

        reward = -shifters
        terminated = (self.current_vessel_slot is None) or (self.available_groups.size == 0)

        valid_actions = self._get_valid_yard_actions() if not terminated else []

        observation = self._create_observation()
        self.total_shifters += shifters
        info.update({
            "shifters": shifters,
            "vessel_slots_filled": self.vessel_slots_filled,
            "total_shifters": self.total_shifters,
            "current_time": self.current_time,
            "action_mask": self._update_action_mask()
        })

        self.current_time += 1

        return observation, reward, terminated, truncated, info

    def _get_shifters(self, action):
        original_bay = self.yard_state[action, StateIds.BAY.value]
        original_row = self.yard_state[action, StateIds.ROW.value]
        original_tier = self.yard_state[action, StateIds.TIER.value]

        same_bay_row_mask = (
                (self.yard_state[:, StateIds.BAY.value] == original_bay)
                & (self.yard_state[:, StateIds.ROW.value] == original_row)
                & (self.yard_state[:, StateIds.TIER.value] > original_tier)
                & (self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1)
        )

        upper_slots = np.where(same_bay_row_mask)[0]
        return upper_slots

    def _process_shifters(self, action, vessel_slot):

        # print("=== BEFORE SHIFT ===")
        # print(self.yard_state.copy())

        # print("yard_state[action] =", self.yard_state[action].copy())

        # print(action)
        # Get coords of selected container, then get all containers above it
        upper_slots = self._get_shifters(action)
        sorted_upper_slots = []

        if len(upper_slots) > 0:
            # Sort containers from bottom to top (ascending tier order)
            sorted_upper_slots = sorted(upper_slots, key=lambda x: self.yard_state[x, StateIds.TIER.value])
            new_yard_state = self.yard_state.copy()
            container_group = self.yard_state[action, StateIds.GROUP.value]

            # Remove the selected container from the yard
            new_yard_state[action, StateIds.IS_OCCUPIED.value] = 0

            # Process containers from bottom to top to maintain gravity constraints
            for slot in sorted_upper_slots:
                current_tier = self.yard_state[slot, StateIds.TIER.value]
                src_idx = slot
                target_bay = self.yard_state[slot, StateIds.BAY.value]
                target_row = self.yard_state[slot, StateIds.ROW.value]
                target_tier = current_tier - 1

                # Find corresponding slot one tier below
                target_idx = np.where(
                    (self.yard_state[:, StateIds.BAY.value] == target_bay)
                    & (self.yard_state[:, StateIds.ROW.value] == target_row)
                    & (self.yard_state[:, StateIds.TIER.value] == target_tier)
                )[0][0]

                # Move container down
                new_yard_state[target_idx, StateIds.IS_OCCUPIED.value] = 1
                new_yard_state[target_idx, StateIds.GROUP.value] = self.yard_state[slot, StateIds.GROUP.value]
                new_yard_state[src_idx, StateIds.IS_OCCUPIED.value] = 0

            self.yard_state = new_yard_state
        else:
            # No containers above, just remove the container and get its group
            container_group = self.yard_state[action, StateIds.GROUP.value]
            self.yard_state[action, StateIds.IS_OCCUPIED.value] = 0

        # Fill the target vessel slot
        self.vessel_state[vessel_slot, StateIds.IS_OCCUPIED.value] = 1
        self.vessel_state[vessel_slot, StateIds.GROUP.value] = container_group
        # Update global information
        self.available_groups = np.unique(
            self.yard_state[self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1, StateIds.GROUP.value]
        )
        self.vessel_slots_filled += 1

        shifters = len(sorted_upper_slots)

        # print("=== after ===")
        # print(self.yard_state.copy())
        return shifters

    def action_masks(self):
        """For compatibility with SB3"""
        if self.current_vessel_slot is None:
            return [False] * self.action_space.n

        valid_actions = self._get_valid_yard_actions()

        return [action in valid_actions for action in range(self.action_space.n)]

    def _update_action_mask(self):
        valid_actions = self._get_valid_yard_actions()

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[valid_actions] = 1

        return mask

    def reset(self, seed=None, **kwargs):
        self._reset()

        # Create observation
        observation = self._create_observation()
        info = {}
        info["action_mask"] = self._update_action_mask()

        return observation, info

    def _reset(self):
        # seed args for compatibility with some RL frameworks, need to remove after test SB3 compatibility
        self.total_shifters = 0
        self.total_timesteps = 0
        yard_bays = self._generate_bay_coords(self.yard_shape[0])
        _, R, T = self.yard_shape
        # print(yard_bays)
        # print(self.total_yard_coords)
        # print(self.num_slot_attrs)
        self.yard_state = np.zeros((self.total_yard_coords, self.num_slot_attrs), dtype=int)
        # print(self.yard_state.shape)  # (300,5)
        # Generate coords in bay-row-tier order
        # To generate all possible bay-row-tier combinations, we can try grouping like this:
        # Starting grouping varied tiers into rows as: row1 -> tier1, tier2, tier3, row2 -> tier1, tier2, tier3...
        # Then grouping rows into bays as: bay1 -> row1(including 3 tiers), row2(incl. 3 tiers), bay2 -> row1, row2...
        # Then we need tiers as num_bays * num_rows * (tier1, tier2, tier3), which can be achieved by using np.tile
        # For rows we need to repeat the rows for each bay, we also need to repeat each element for a whole tier array
        # like num_bays * (row1, row1, row1, row2, row2, row2, ...). This can be achieved by using np.tile for outer array
        # and np.repeat for inner array
        # For bays we just need to repeat each elements num_rows * num_tiers times
        self.yard_state[:, StateIds.BAY.value] = np.repeat(yard_bays, R * T)
        self.yard_state[:, StateIds.ROW.value] = np.tile(np.arange(1, R + 1).repeat(T), len(yard_bays))
        self.yard_state[:, StateIds.TIER.value] = np.tile(np.arange(1, T + 1), len(yard_bays) * R)

        # print(self.yard_state)

        vessel_bays = self._generate_bay_coords(self.vessel_shape[0])
        _, Rv, Tv = self.vessel_shape
        self.vessel_state = np.zeros((self.total_vessel_coords, self.num_slot_attrs), dtype=int)

        self.vessel_state[:, StateIds.BAY.value] = np.repeat(vessel_bays, Rv * Tv)
        self.vessel_state[:, StateIds.ROW.value] = np.tile(np.arange(1, Rv + 1).repeat(Tv), len(vessel_bays))
        self.vessel_state[:, StateIds.TIER.value] = np.tile(np.arange(1, Tv + 1), len(vessel_bays) * Rv)

        # print(f"vessel_state{self.vessel_state}")

        if self.group_num > 1:
            odd_bay_mask = self.vessel_state[:, StateIds.BAY.value] % 2 == 1

            odd_bay_indices = np.where(odd_bay_mask)[0]
            # print(odd_bay_indices)
            valid_slots = odd_bay_indices

            # Assign groups to vessel slots
            slots_per_group = len(valid_slots) // self.group_num

            # print(slots_per_group)  每个组可以分到多少槽位

            for group in range(self.group_num):
                start_pos = group * slots_per_group
                end_pos = (group + 1) * slots_per_group if group < self.group_num - 1 else len(valid_slots)
                group_indices = valid_slots[start_pos:end_pos]
                if len(group_indices) > 0:
                    self.vessel_state[group_indices, StateIds.GROUP.value] = group

        # print(self.vessel_state) 现在的 vessel 有组了

        self.vessel_slots_filled = 0  # 当前 episode 中，已经成功往船上放了多少个箱子

        self._initialize_containers()

        # print(f"self.yard_state{self.yard_state}")

        self.available_groups = np.unique(
            self.yard_state[self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1, StateIds.GROUP.value]
        )  # 获取当前所有被占用 slot 所属的 group 编号（去重）

        # print(self.available_groups)
        self.current_vessel_slot = self._get_next_vessel_slot()

    def _generate_bay_coords(self, physical_bays: int) -> list:

        """Generate bay coordinates based on the number of physical bays. Example: 4 physical bays will have bay_groups as [1, 2, 3], [5]"""
        bay_groups = []
        group_start = 1

        for _ in range(physical_bays // 2):
            bay_groups.extend([group_start, group_start + 1, group_start + 2])
            group_start += 4

        if physical_bays % 2 == 1:
            bay_groups.append(group_start)

        return bay_groups

    def _get_next_vessel_slot(self):

        valid_slots = np.where(
            (self.vessel_state[:, StateIds.IS_OCCUPIED.value] == 0)
            & (self.vessel_state[:, StateIds.BAY.value] % 2 == 1)
            & np.isin(self.vessel_state[:, StateIds.GROUP.value], self.available_groups)
        )[0]
        if len(valid_slots) == 0:
            return None

        # print(valid_slots)
        # Retrieve the bay, row, and tier coordinates for the valid slots.
        bays = self.vessel_state[valid_slots, StateIds.BAY.value]
        rows = self.vessel_state[valid_slots, StateIds.ROW.value]
        tiers = self.vessel_state[valid_slots, StateIds.TIER.value]

        # print(valid_slots)
        sort_order = np.lexsort((rows, tiers, bays))  # sort_order 是位置索引   # 这个还有点没看懂
        return valid_slots[sort_order[0]]  # vessel_state 的行索引（slot 编号） # 这个还有点没看懂

    def _create_observation(self):

        state = np.concatenate((self.vessel_state, self.yard_state), axis=0)

        if self.current_vessel_slot is not None:
            state = np.concatenate((state, self.vessel_state[self.current_vessel_slot].reshape(1, -1)), axis=0)
        else:
            state = np.concatenate((state, np.zeros((1, 5), dtype=int)), axis=0)

        # print(f"state{state.shape}")
        state = state.flatten()

        # print(state.shape)
        if self.action_mask == "default":
            return state
        else:
            mask = self.action_masks()
            return {"observation": state, "mask": mask}

    def _get_valid_yard_actions(self) -> np.ndarray:
        if self.current_vessel_slot is None:
            return []
        occupied_mask = self.yard_state[:, StateIds.IS_OCCUPIED.value] == 1  # 真正有箱子的slot

        bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 == 1  # 只允许奇数bay

        # yard 中箱子的 group 必须等于当前要填的 船舱位的 group
        group_mask = (
                self.yard_state[:, StateIds.GROUP.value]
                == self.vessel_state[self.current_vessel_slot, StateIds.GROUP.value]
        )

        valid_actions = np.where(occupied_mask & bay_mask & group_mask)[0]
        return valid_actions

    def _initialize_containers(self):
        odd_bay_mask = self.yard_state[:, StateIds.BAY.value] % 2 != 0
        available_slots = np.where(odd_bay_mask)[0]
        # print(len(available_slots)) # 200
        num_to_set = min(self.num_containers, len(available_slots))

        # print(f"self.num_containers{self.num_containers}")
        # print(f"len(available_slots){len(available_slots)}")

        if num_to_set > 0:
            selected_slots = available_slots[:num_to_set]
            self.yard_state[selected_slots, StateIds.IS_OCCUPIED.value] = 1

            if self.group_num > 1:
                containers_per_group = num_to_set // self.group_num

                if self.group_placement == "fixed":
                    for group in range(self.group_num):
                        start_idx = group * containers_per_group
                        end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                        if start_idx < end_idx:
                            self.yard_state[selected_slots[start_idx:end_idx], StateIds.GROUP.value] = group
                else:
                    rng = np.random.RandomState(self.seed)

                    shuffled_indices = rng.permutation(num_to_set)
                    shuffled_slots = selected_slots[shuffled_indices]
                    for group in range(self.group_num):
                        start_idx = group * containers_per_group
                        end_idx = (group + 1) * containers_per_group if group < self.group_num - 1 else num_to_set

                        if start_idx < end_idx:
                            self.yard_state[shuffled_slots[start_idx:end_idx], StateIds.GROUP.value] = group
        print(self.yard_state)

    def render(self):
        """Render the environment as an RGB array"""
        if self.render_mode != "rgb_array":
            return None

        try:
            import os

            os.environ["SDL_VIDEODRIVER"] = "dummy"
            import pygame
        except ImportError:
            raise ImportError("pygame is not installed")

        if not pygame.get_init():
            pygame.init()
        if self.screen is None:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.screen.fill((255, 255, 255))

        padding, title_height, section_gap = 10, 20, 50
        vessel_height = (self.screen_height - 3 * padding - 2 * title_height) * 0.4
        yard_height = (self.screen_height - 3 * padding - 2 * title_height) * 0.6

        font = pygame.font.Font(None, 24)
        self.screen.blit(font.render("Vessel", True, (0, 0, 0)), (padding, padding))
        self.screen.blit(font.render("Yard", True, (0, 0, 0)), (padding, padding + vessel_height + section_gap))

        self._draw_grid(
            self.vessel_state,
            padding + title_height,
            vessel_height,
            self.vessel_shape[0],
            self.vessel_shape[1],
            self.vessel_shape[2],
            True,
        )
        self._draw_grid(
            self.yard_state,
            3 * padding + 2 * title_height + vessel_height + section_gap,
            yard_height,
            self.yard_shape[0],
            self.yard_shape[1],
            self.yard_shape[2],
            False,
        )

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _draw_grid(self, state, top, height, bays, rows, tiers, is_vessel):
        """Draw a grid section (vessel or yard) with fixed cell size"""
        import pygame

        cell_width, cell_height, left_margin, label_margin = self._setup_grid_dimensions(bays, rows)
        fonts = {"small": pygame.font.Font(None, 20), "tiny": pygame.font.Font(None, 18)}
        colors = self._setup_colors()
        self._draw_tier_labels(top, tiers, cell_height, left_margin, label_margin, fonts["small"])
        row_order = self._get_row_order(rows, is_vessel)
        cell_info = self._build_cell_info(state, bays, is_vessel)
        self._draw_grid_cells(
            cell_info,
            top,
            left_margin,
            bays,
            rows,
            tiers,
            row_order,
            cell_width,
            cell_height,
            label_margin,
            colors,
            fonts,
            is_vessel,
        )
        self._draw_bay_dividers(left_margin, top, bays, rows, cell_width, tiers, cell_height, colors["bay_grid"])

    def _setup_grid_dimensions(self, bays, rows):
        cell_width, cell_height = 35, 35
        padding, label_margin = 30, 15
        left_margin = max(padding, (self.screen_width - bays * rows * cell_width) / 2)
        return cell_width, cell_height, left_margin, label_margin

    def _setup_colors(self):
        group_colors = []
        for i in range(self.group_num):
            hue = i / self.group_num
            r, g, b = colorsys.hsv_to_rgb(hue, 0.3, 0.95)
            light_color = (int(r * 255), int(g * 255), int(b * 255))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.7)
            dark_color = (int(r * 255), int(g * 255), int(b * 255))
            group_colors.append((light_color, dark_color))

        return {
            "group_colors": group_colors,
            "empty": (255, 255, 255),
            "thin_grid": (180, 180, 180),
            "bay_grid": (0, 0, 0),
            "target": (255, 0, 0),
        }

    def _get_row_order(self, rows, is_vessel):
        """Determine row ordering based on vessel or yard"""
        if is_vessel:
            if rows % 2 == 0:  # Even number of rows
                left = list(range(rows - 1, 0, -2))
                right = list(range(2, rows + 1, 2))
                return left + right
            else:  # Odd number of rows
                left = list(range(rows, 0, -2))
                right = list(range(2, rows, 2))
                return left + right
        else:
            return list(range(1, rows + 1))

    def _draw_tier_labels(self, top, tiers, cell_height, left_margin, label_margin, font):
        """Draw tier labels on the left side"""
        for t in range(1, tiers + 1):
            y = top + (tiers - t) * cell_height + cell_height / 2
            tier_label = font.render(f"{t}", True, (0, 0, 0))
            self.screen.blit(tier_label, (left_margin - label_margin, y - tier_label.get_height() / 2))

    def _build_cell_info(self, state, bays, is_vessel):
        """Build cell information dictionary for rendering"""
        cell_info = {}

        if is_vessel:
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                is_target = self._is_target_cell(i, is_vessel)
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1:
                    cell_info[(bay, row, tier)] = self._create_cell_props(is_occupied, is_target, i, group)

            # handle even bays
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                # Only apply even bay influence if it's occupied
                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            existing = cell_info.get(
                                (adj_bay, row, tier), self._create_cell_props(False, False, None, group)
                            )
                            existing["filled"] = True
                            existing["idx"] = i
                            cell_info[(adj_bay, row, tier)] = existing
        else:
            # Handle yard containers
            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 1 and is_occupied:
                    cell_info[(bay, row, tier)] = self._create_cell_props(True, False, i, group)

            for i in range(len(state)):
                bay = int(state[i, StateIds.BAY.value])
                row = int(state[i, StateIds.ROW.value])
                tier = int(state[i, StateIds.TIER.value])
                is_occupied = state[i, StateIds.IS_OCCUPIED.value] == 1
                group = int(state[i, StateIds.GROUP.value])

                if bay % 2 == 0 and is_occupied:
                    for adj_bay in [bay - 1, bay + 1]:
                        if 1 <= adj_bay <= bays * 2:
                            cell_info[(adj_bay, row, tier)] = self._create_cell_props(True, False, i, group)

        return cell_info

    def _is_target_cell(self, idx, is_vessel):
        if is_vessel:
            return self.current_vessel_slot == idx if self.current_vessel_slot is not None else False
        return False

    def _create_cell_props(self, filled, target, idx, group):
        # Used for inheritance
        return {"filled": filled, "target": target, "idx": idx, "group": group}

    def _get_cell_border_style(self, cell):
        if cell["target"]:
            return (255, 0, 0), 3  # Red border for target
        return (180, 180, 180), 1

    def _draw_grid_cells(
            self,
            cell_info,
            top,
            left_margin,
            bays,
            rows,
            tiers,
            row_order,
            cell_width,
            cell_height,
            label_margin,
            colors,
            fonts,
            is_vessel,
    ):
        """Draw grid cells with labels"""
        for b in range(bays):
            bay_num = b * 2 + 1
            bay_x = left_margin + b * rows * cell_width

            bay_center_x = bay_x + (rows * cell_width) / 2
            bay_label = fonts["small"].render(f"Bay {bay_num}", True, (0, 0, 0))
            self.screen.blit(bay_label, bay_label.get_rect(center=(bay_center_x, top - 10)))

            for pos, r in enumerate(row_order):
                # Draw row label
                x_label = bay_x + pos * cell_width + cell_width / 2
                row_label = fonts["small"].render(f"{r}", True, (0, 0, 0))
                self.screen.blit(
                    row_label, row_label.get_rect(center=(x_label, top + tiers * cell_height + label_margin / 2))
                )

                # Draw cells for each tier
                for t in range(1, tiers + 1):
                    x = bay_x + pos * cell_width
                    y = top + (tiers - t) * cell_height

                    default_cell = self._create_cell_props(False, False, None, 0)
                    cell = cell_info.get((bay_num, r, t), default_cell)

                    self._draw_cell(x, y, cell_width, cell_height, cell, colors, fonts, is_vessel)

    def _draw_cell(self, x, y, width, height, cell, colors, fonts, is_vessel):
        """Draw an individual cell"""
        import pygame

        group_colors = colors["group_colors"]
        group_idx = min(cell["group"], len(group_colors) - 1)

        # Determine fill color
        if cell["filled"]:
            color = group_colors[group_idx][1]
        elif is_vessel:
            color = group_colors[group_idx][0]
        else:
            color = colors["empty"]

        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, rect)

        line_color, line_width = self._get_cell_border_style(cell)
        pygame.draw.rect(self.screen, line_color, rect, line_width)
        if cell["idx"] is not None:
            if is_vessel:
                text_color = (255, 255, 255) if cell["filled"] else (50, 50, 50)
                label = fonts["tiny"].render(f"{cell['idx']}", True, text_color)
                self.screen.blit(label, label.get_rect(center=(x + width / 2, y + height / 2)))
            elif cell["filled"]:
                label = fonts["tiny"].render(f"{cell['idx']}", True, (255, 255, 255))
                self.screen.blit(label, label.get_rect(center=(x + width / 2, y + height / 2)))

    def _draw_bay_dividers(self, left_margin, top, bays, rows, cell_width, tiers, cell_height, color):
        """Draw vertical divider lines between bays"""
        import pygame

        for b in range(bays + 1):
            x = left_margin + b * rows * cell_width
            pygame.draw.line(
                self.screen,
                color,
                (x, top),
                (x, top + tiers * cell_height),
                2,
            )

    def _get_randomized_time_array(self):
        rng = np.random.RandomState(self.seed)
        return rng.randint(low=1, high=100, size=self.action_space.n, dtype=np.int32)
