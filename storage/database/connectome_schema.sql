create table glio_vascular
(
    guid                         integer          not null,
    astrocyte_population_guid    integer          not null,
    vasculature_population_guid  integer          not null,
    astrocyte_guid               integer          not null,
    astrocyte_section_guid       integer          not null,
    vasculature_node_guid        integer          not null,
    vasculature_section_guid     integer          not null,
    vasculature_segment_guid     integer          not null,
    endfoot_compartment_length   double precision not null,
    endfoot_compartment_diameter double precision not null,
    constraint glio_vascular_pk
        primary key (guid, astrocyte_population_guid, vasculature_population_guid)
)
    tablespace bioexplorer_ts;

alter table glio_vascular
    owner to bioexplorer;

create index glio_vascular_astrocyte_guid_index
    on glio_vascular (astrocyte_guid)
    tablespace bioexplorer_ts;

create unique index glio_vascular_astrocyte_section_guid_vasculature_section_guid_v
    on glio_vascular (astrocyte_section_guid, vasculature_section_guid, vasculature_segment_guid)
    tablespace bioexplorer_ts;

create index glio_vascular_vasculature_node_guid_astrocyte_guid_index
    on glio_vascular (vasculature_node_guid, astrocyte_guid)
    tablespace bioexplorer_ts;

create index glio_vascular_vasculature_node_guid_index
    on glio_vascular (vasculature_node_guid)
    tablespace bioexplorer_ts;

create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    schema      varchar,
    description varchar
)
    tablespace bioexplorer_ts;

alter table population
    owner to bioexplorer;

create table structure
(
    guid        integer not null
        constraint structure_pk
            primary key,
    code        varchar not null,
    description varchar
)
    tablespace bioexplorer_ts;

alter table structure
    owner to bioexplorer;

create table streamline
(
    guid               integer not null,
    hemisphere_guid    integer not null,
    origin_region_guid integer not null
        constraint streamline_structure_guid_fk
            references structure
            on update cascade on delete cascade,
    target_region_guid integer not null
        constraint streamline_structure_guid_fk_2
            references structure
            on update cascade on delete cascade,
    points             bytea   not null
)
    tablespace bioexplorer_ts;

alter table streamline
    owner to bioexplorer;

create unique index streamline_guid_origin_region_guid_target_region_guid_uindex
    on streamline (guid, origin_region_guid, target_region_guid)
    tablespace bioexplorer_ts;

create unique index structure_guid_uindex
    on structure (guid)
    tablespace bioexplorer_ts;

