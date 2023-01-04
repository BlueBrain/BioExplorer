-- The Blue Brain BioExplorer is a tool for scientists to extract and analyse
-- scientific data from visualization
--
-- Copyright 2020-2023 Blue BrainProject / EPFL
--
-- This program is free software: you can redistribute it and/or modify it under
-- the terms of the GNU General Public License as published by the Free Software
-- Foundation, either version 3 of the License, or (at your option) any later
-- version.
--
-- This program is distributed in the hope that it will be useful, but WITHOUT
-- ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
-- FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
-- details.
--
-- You should have received a copy of the GNU General Public License along with
-- this program.  If not, see <https://www.gnu.org/licenses/>.

CREATE SCHEMA IF NOT EXISTS neurons;
CREATE SCHEMA IF NOT EXISTS astrocytes;
CREATE SCHEMA IF NOT EXISTS vasculature;
CREATE SCHEMA IF NOT EXISTS connectome;
CREATE SCHEMA IF NOT EXISTS atlas;


-- Create Public utility functions
create function length(anyarray) returns double precision
    immutable
    strict
    language sql
as
$$
select sqrt($1[1] * $1[1] + $1[2] * $1[2] + $1[3] * $1[3]);
$$;

create function norm(anyarray) returns anyarray
    immutable
    strict
    language sql
as
$$
select array[$1[1] / length($1), $1[2] / length($1), $1[3] / length($1)];
$$;

create function "cross"(anyarray, anyarray) returns anyarray
    immutable
    strict
    language sql
as
$$
select array[$1[2] * $2[3] - $1[3] * $2[2], $1[3] * $2[1] - $1[1] * $2[3], $1[1] * $2[2] - $1[2] * $2[1]];
$$;
